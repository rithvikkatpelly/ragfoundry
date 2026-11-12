import asyncio
import importlib.util

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ragfoundry import ConfigurationError, RAGConfig, RAGPipeline, pack_documents, window_history
from ragfoundry.config import ContextWindowConfig
from ragfoundry.context import (
    approximate_tokens,
    build_token_counter,
    format_context,
    normalize_history,
)

from .conftest import KeywordEmbeddings, rule_llm


def words(text: str) -> int:
    return len(text.split())


def docs(*texts: str) -> list[Document]:
    return [Document(page_content=t, metadata={"source": f"d{i}.md"}) for i, t in enumerate(texts)]


# ----------------------------------------------------------------------- packing
def test_pack_without_budget_keeps_everything():
    packed = pack_documents(docs("a b", "c d"), None, words)
    assert len(packed.documents) == 2
    assert packed.tokens == words(format_context(packed.documents))
    assert (packed.dropped, packed.truncated) == (0, False)


def test_pack_stops_at_budget_and_trims_the_overflowing_passage():
    long = " ".join(f"w{i}" for i in range(200))
    items = docs("one two three", long, "never reached")
    packed = pack_documents(items, 60, words)

    assert [d.metadata["source"] for d in packed.documents] == ["d0.md", "d1.md"]
    assert packed.tokens <= 60
    assert packed.tokens == words(format_context(packed.documents))
    trimmed = packed.documents[1]
    assert trimmed.metadata["truncated"] is True
    assert trimmed.page_content.endswith("…")
    assert long.startswith(trimmed.page_content[:-2].rstrip())
    assert "truncated" not in items[1].metadata  # originals untouched
    assert packed.dropped == 1 and packed.truncated


def test_pack_drops_instead_of_trimming_when_disabled():
    long = " ".join(f"w{i}" for i in range(200))
    packed = pack_documents(docs("one two", long), 60, words, truncate_last=False)
    assert len(packed.documents) == 1
    assert packed.dropped == 1 and not packed.truncated


def test_pack_skips_tiny_remainders():
    packed = pack_documents(docs("x " * 20, "y " * 50), 30, words)
    # 20 words + header leave fewer than 16 tokens, so the second passage is dropped whole.
    assert len(packed.documents) == 1


def test_approximate_counter():
    assert approximate_tokens("") == 0
    assert approximate_tokens("abcd") == 1
    assert approximate_tokens("abcde") == 2


@pytest.mark.skipif(importlib.util.find_spec("tiktoken") is None, reason="tiktoken not installed")
def test_tiktoken_counter():
    count = build_token_counter(ContextWindowConfig(token_counter="tiktoken"))
    assert count("hello world") == 2


# ----------------------------------------------------------------------- history
def conversation(turns: int) -> list:
    messages = []
    for i in range(turns):
        messages += [HumanMessage(content=f"question {i}"), AIMessage(content=f"answer {i}")]
    return messages


def test_window_keeps_latest_whole_turns():
    kept = window_history(conversation(6), 2, None, words)
    assert [m.content for m in kept] == ["question 4", "answer 4", "question 5", "answer 5"]
    assert window_history(conversation(3), 0, None, words) == []


def test_window_token_budget_drops_oldest_turns():
    kept = window_history(conversation(5), 10, 9, words)  # each turn is 4 words
    assert [m.content for m in kept] == ["question 3", "answer 3", "question 4", "answer 4"]


def test_normalize_history_formats():
    messages = normalize_history(
        [
            {"role": "user", "content": "hi"},
            ("assistant", "hello"),
            HumanMessage(content="next"),
            SystemMessage(content="ignored"),
        ]
    )
    assert [(m.type, m.content) for m in messages] == [
        ("human", "hi"),
        ("ai", "hello"),
        ("human", "next"),
    ]
    with pytest.raises(ConfigurationError, match="role"):
        normalize_history([{"role": "robot", "content": "x"}])
    with pytest.raises(ConfigurationError, match="Unsupported"):
        normalize_history(["just a string"])


# ---------------------------------------------------------------------- pipeline
def chat_responder(prompt: str) -> str:
    if "Follow-up question:" in prompt:
        if "solar" in prompt and "efficient is it" in prompt:
            return "How efficient are solar panels?"
        return prompt.rsplit("Follow-up question:", 1)[1].strip()
    if "Context:" in prompt:
        return "Silicon cells are about 20 percent efficient [1]."
    return "no"


def make_pipeline(tmp_path, llm=None, **settings):
    data = {"persist_dir": str(tmp_path / "store"), "vector_store": {"provider": "chroma"}}
    data.update(settings)
    return RAGPipeline(
        RAGConfig.model_validate(data),
        llm=llm or rule_llm(chat_responder),
        embeddings=KeywordEmbeddings(),
    )


def test_chat_session_condenses_follow_ups_and_sends_history(tmp_path, corpus):
    llm = rule_llm(chat_responder)
    with make_pipeline(tmp_path, llm) as rag:
        rag.ingest(corpus)
        chat = rag.chat()

        first = chat.ask("Tell me about solar panels")
        assert first.standalone_question is None
        assert len(llm.prompts) == 1  # no history yet, so no condensing call

        follow_up = chat.ask("How efficient is it?")
        assert follow_up.standalone_question == "How efficient are solar panels?"
        assert follow_up.sources[0].endswith("solar.md")
        generation_prompt = llm.prompts[-1]
        assert "Tell me about solar panels" in generation_prompt  # history reached the LLM
        assert "Question: How efficient is it?" in generation_prompt  # original wording kept
        assert len(chat) == 2
        assert chat.to_dicts()[0] == {"role": "user", "content": "Tell me about solar panels"}

        resumed = rag.chat(chat.to_dicts())
        assert len(resumed) == 2
        chat.reset()
        assert len(chat) == 0


def test_history_window_limits_what_is_sent(tmp_path, corpus):
    llm = rule_llm(chat_responder)
    window = {"max_history_turns": 1, "condense_question": False}
    with make_pipeline(tmp_path, llm, context_window=window) as rag:
        rag.ingest(corpus)
        history = conversation(3)
        result = rag.query("solar panel efficiency", history=history)
        prompt = llm.prompts[-1]
        assert "question 2" in prompt and "question 1" not in prompt
        assert result.standalone_question is None
        assert len(llm.prompts) == 1  # condensing disabled


def test_context_budget_applies_to_generation(tmp_path, corpus):
    with make_pipeline(tmp_path, context_window={"max_context_tokens": 40}) as rag:
        rag.ingest(corpus)
        unlimited = rag.retrieve("solar panels sunlight sourdough starter part")
        result = rag.query("solar panels sunlight sourdough starter part")
        assert result.context_tokens <= 40
        assert 0 < len(result.documents) < len(unlimited)
        assert result.to_dict()["context_tokens"] == result.context_tokens


def test_custom_token_counter(tmp_path, corpus):
    calls = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    data = {"persist_dir": str(tmp_path / "s"), "vector_store": {"provider": "chroma"}}
    rag = RAGPipeline(
        RAGConfig.model_validate({**data, "context_window": {"max_context_tokens": 500}}),
        llm=rule_llm(chat_responder),
        embeddings=KeywordEmbeddings(),
        token_counter=counter,
    )
    with rag:
        rag.ingest(corpus)
        result = rag.query("solar panels")
        assert calls
        assert result.context_tokens == counter(format_context(result.documents))


def test_chat_stream_records_turn(tmp_path, corpus):
    with make_pipeline(tmp_path) as rag:
        rag.ingest(corpus)
        chat = rag.chat()
        text = "".join(chat.stream("Tell me about solar panels"))
        assert text == "Silicon cells are about 20 percent efficient [1]."
        assert chat.to_dicts()[1] == {"role": "assistant", "content": text}

        async def go() -> str:
            result = await chat.aask("How efficient is it?")
            return result.standalone_question or ""

        assert asyncio.run(go()) == "How efficient are solar panels?"
        assert len(chat) == 2


def test_retrieval_needs_no_llm_without_history(tmp_path, corpus):
    config = RAGConfig.model_validate(
        {"persist_dir": str(tmp_path / "s"), "vector_store": {"provider": "chroma"}}
    )
    with RAGPipeline(config, embeddings=KeywordEmbeddings()) as rag:
        rag.ingest(corpus)
        assert rag.retrieve("solar panels")[0].metadata["source"].endswith("solar.md")
        with pytest.raises(ConfigurationError, match="No LLM"):
            rag.query("solar panels")


def test_context_window_yaml(tmp_path):
    path = tmp_path / "rag.yaml"
    path.write_text(
        "context_window:\n  max_context_tokens: 3000\n  max_history_turns: 3\n"
        "  token_counter: tiktoken\n"
    )
    window = RAGConfig.from_yaml(path).context_window
    assert (window.max_context_tokens, window.max_history_turns) == (3000, 3)
    assert window.condense_question is True
    with pytest.raises(ValueError):
        RAGConfig.model_validate({"context_window": {"max_context_tokens": 0}})
