import json

import pytest
from langchain_core.documents import Document

from ragfoundry import ConfigurationError, chunk_documents
from ragfoundry.chunking import ChunkerResources, LLMChunker, SemanticChunker, assign_chunk_ids
from ragfoundry.config import ChunkingConfig

from .conftest import rule_llm

TEXT = " ".join(f"Sentence number {i} talks about item {i}." for i in range(60))


def test_fixed_window_size_and_overlap():
    config = ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=20)
    chunks = chunk_documents([Document(page_content=TEXT, metadata={"source": "s"})], config)
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert all(len(c.page_content) == 100 for c in chunks[:-1])
    assert chunks[0].page_content[-20:] == chunks[1].page_content[:20]
    assert chunks[-1].page_content.endswith(TEXT[-10:])
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        assert TEXT[start : start + len(chunk.page_content)] == chunk.page_content


def test_token_length_unit():
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    for strategy in ("fixed", "recursive"):
        config = ChunkingConfig(
            strategy=strategy, chunk_size=40, chunk_overlap=5, length_unit="tokens"
        )
        chunks = chunk_documents([Document(page_content=TEXT)], config)
        assert len(chunks) > 1
        assert all(len(encoding.encode(c.page_content)) <= 40 for c in chunks)


def test_recursive_prefers_paragraph_breaks():
    text = "First paragraph about cats.\n\nSecond paragraph about dogs."
    config = ChunkingConfig(chunk_size=40, chunk_overlap=0)
    chunks = chunk_documents([Document(page_content=text)], config)
    assert [c.page_content for c in chunks] == [
        "First paragraph about cats.",
        "Second paragraph about dogs.",
    ]


def test_semantic_splits_at_topic_change(embeddings):
    text = (
        "Solar panels convert sunlight. Solar panels need sunlight. Sunlight powers panels. "
        "Sourdough needs flour starter. Flour starter feeds sourdough. Sourdough flour bakes."
    )
    splitter = SemanticChunker(
        embeddings, breakpoint_percentile=80, chunk_size=1000, chunk_overlap=0
    )
    assert splitter.split_text(text) == [
        "Solar panels convert sunlight. Solar panels need sunlight. Sunlight powers panels.",
        "Sourdough needs flour starter. Flour starter feeds sourdough. Sourdough flour bakes.",
    ]


def test_semantic_enforces_maximum_size(embeddings):
    splitter = SemanticChunker(
        embeddings, breakpoint_percentile=99, chunk_size=120, chunk_overlap=0
    )
    assert all(len(chunk) <= 120 for chunk in splitter.split_text(TEXT))


def _sentence_numbers(prompt: str) -> list[int]:
    return [int(line[1 : line.index("]")]) for line in prompt.splitlines() if line.startswith("[")]


def test_llm_chunker_uses_model_boundaries():
    text = "Cats purr. Cats nap. Dogs bark. Dogs fetch. Fish swim."
    llm = rule_llm(lambda prompt: "Boundaries: [2, 4]")
    chunks = LLMChunker(llm, chunk_size=500, chunk_overlap=0).split_text(text)
    assert chunks == ["Cats purr. Cats nap.", "Dogs bark. Dogs fetch.", "Fish swim."]


def test_llm_chunker_carries_last_group_between_windows():
    text = " ".join(f"Topic {i // 4} sentence {i}." for i in range(12))

    def responder(prompt: str) -> str:
        numbers = _sentence_numbers(prompt)
        return json.dumps([n for n in numbers[1:] if n % 4 == 0])

    llm = rule_llm(responder)
    chunks = LLMChunker(llm, window_chars=130, chunk_size=1000, chunk_overlap=0).split_text(text)
    assert len(llm.prompts) > 1, "the text must span several windows for this test"
    assert [chunk.count("sentence") for chunk in chunks] == [4, 4, 4]


def test_llm_chunker_falls_back_on_unusable_reply(caplog):
    llm = rule_llm(lambda prompt: "I cannot help with that.")
    chunks = LLMChunker(llm, chunk_size=200, chunk_overlap=0).split_text(TEXT)
    assert chunks and all(len(c) <= 200 for c in chunks)
    assert "Unusable LLM chunking reply" in caplog.text


def test_llm_strategy_needs_an_llm():
    with pytest.raises(ConfigurationError, match="requires an LLM"):
        chunk_documents([Document(page_content=TEXT)], ChunkingConfig(strategy="llm"))
    llm = rule_llm(lambda prompt: "[]")
    chunks = chunk_documents(
        [Document(page_content="One. Two.")],
        ChunkingConfig(strategy="llm"),
        ChunkerResources(llm=lambda: llm),
    )
    assert [c.page_content for c in chunks] == ["One. Two."]


def test_chunk_ids_are_stable_and_distinguish_duplicates():
    def chunks(*texts):
        return assign_chunk_ids([Document(page_content=t, metadata={"source": "f"}) for t in texts])

    first = [c.metadata["chunk_id"] for c in chunks("a", "b", "a")]
    second = [c.metadata["chunk_id"] for c in chunks("new", "a", "b", "a")]
    assert len(set(first)) == 3
    assert second[1:] == first
