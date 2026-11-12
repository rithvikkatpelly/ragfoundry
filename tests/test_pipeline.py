import asyncio
import re

import pytest
from langchain_core.documents import Document

from ragfoundry import ConfigurationError, RAGConfig, RAGPipeline
from ragfoundry.retrieval import reciprocal_rank_fusion

from .conftest import KeywordEmbeddings, rule_llm


def answer_from_context(prompt: str) -> str:
    if "Context:" not in prompt:
        return "no"
    first = re.search(r"\[1\] \(source: [^)]*\)\n(.+)", prompt)
    return f"Based on the documents: {first.group(1)[:60]} [1]"


def make_pipeline(tmp_path, provider="faiss", llm=None, embeddings=None, **settings):
    data = {"persist_dir": str(tmp_path / "store"), "vector_store": {"provider": provider}}
    data.update(settings)
    return RAGPipeline(
        RAGConfig.model_validate(data),
        llm=llm or rule_llm(answer_from_context),
        embeddings=embeddings or KeywordEmbeddings(),
    )


@pytest.mark.parametrize("provider", ["faiss", "chroma", "qdrant"])
def test_ingest_query_and_filter(tmp_path, corpus, provider):
    with make_pipeline(tmp_path, provider) as pipeline:
        report = pipeline.ingest(corpus)
        assert report.chunks_added == report.chunks_total == 4
        result = pipeline.query("How efficient are solar panels?")
        assert result.sources[0].endswith("solar.md")
        assert "[1]" in result.answer
        filtered = pipeline.retrieve("panels", where={"file_type": "txt"})
        assert {d.metadata["file_type"] for d in filtered} == {"txt"}


def test_reingest_is_incremental_and_removes_stale_chunks(tmp_path, corpus):
    embeddings = KeywordEmbeddings()
    with make_pipeline(tmp_path, embeddings=embeddings) as pipeline:
        pipeline.ingest(corpus)
        again = pipeline.ingest(corpus)
        assert (again.chunks_added, again.chunks_unchanged, again.chunks_removed) == (0, 4, 0)

        (corpus / "parts.csv").write_text(
            "part_number,description\nXQ-4471,hydraulic pump gasket\n"
        )
        changed = pipeline.ingest(corpus)
        assert (changed.chunks_added, changed.chunks_removed) == (0, 1)
        assert sum(pipeline.sources().values()) == 3
        assert "ZT-9902" not in " ".join(d.page_content for d in pipeline.retrieve("brake caliper"))

    with make_pipeline(tmp_path, embeddings=embeddings) as reopened:
        assert reopened.retrieve("sourdough starter")[0].metadata["source"].endswith("baking.txt")
        removed = reopened.delete_source(str(corpus / "baking.txt"))
        assert removed == 1
        assert reopened.delete_source(str(corpus / "baking.txt")) == 0
        assert not any("baking" in s for s in reopened.sources())


def test_failed_source_keeps_existing_chunks(tmp_path, corpus):
    with make_pipeline(tmp_path, loading={"on_error": "skip"}) as pipeline:
        pipeline.ingest(corpus)
        report = pipeline.ingest([corpus / "solar.md", tmp_path / "missing.pdf"])
        assert len(report.failures) == 1
        assert report.chunks_removed == 0
        assert len(pipeline.sources()) == 3


def test_changing_embedding_model_is_refused(tmp_path, corpus):
    config = {
        "persist_dir": str(tmp_path / "store"),
        "vector_store": {"provider": "faiss"},
        "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
    }
    first = RAGPipeline(RAGConfig.model_validate(config))
    assert first.chunk_store  # binds the collection to this model
    config["embeddings"]["model"] = "text-embedding-3-large"
    with pytest.raises(ConfigurationError, match="text-embedding-3-small"):
        _ = RAGPipeline(RAGConfig.model_validate(config)).chunk_store


def test_hybrid_search_finds_exact_identifiers():
    vector = [Document(page_content="pump", metadata={"chunk_id": n}) for n in "abc"]
    keyword = [Document(page_content="pump", metadata={"chunk_id": n}) for n in "dc"]
    fused = reciprocal_rank_fusion([(vector, 1.0), (keyword, 1.0)])
    assert [d.metadata["chunk_id"] for d in fused][:2] == ["c", "a"]


def test_hybrid_ranks_keyword_match(tmp_path, corpus):
    with make_pipeline(tmp_path, retrieval={"top_k": 2, "hybrid": {"enabled": True}}) as pipeline:
        pipeline.ingest(corpus)
        documents = pipeline.retrieve("XQ-4471")
        assert "XQ-4471" in documents[0].page_content
        assert "rrf_score" in documents[0].metadata


def test_reranker_reorders_and_truncates(tmp_path, corpus):
    class PreferBaking:
        def rerank(self, query, documents, top_n):
            ranked = sorted(documents, key=lambda d: "baking" not in d.metadata["source"])
            return ranked[:top_n]

    pipeline = make_pipeline(tmp_path, retrieval={"top_k": 1})
    pipeline._reranker = PreferBaking()
    pipeline.ingest(corpus)
    documents = pipeline.retrieve("solar panels")
    assert len(documents) == 1
    assert documents[0].metadata["source"].endswith("baking.txt")


def test_query_transforms_feed_retrieval(tmp_path, corpus):
    def responder(prompt):
        if "differently worded" in prompt:
            return '["photovoltaic electricity", "sunlight conversion"]'
        if "factual passage" in prompt:
            return "Photovoltaic cells in panels turn sunlight into electricity."
        if "self-contained search query" in prompt:
            return "solar panel efficiency"
        return answer_from_context(prompt)

    settings = {
        "query_transform": {"strategies": ["rewrite", "multi_query", "hyde"], "num_queries": 2}
    }
    with make_pipeline(tmp_path, llm=rule_llm(responder), retrieval=settings) as pipeline:
        pipeline.ingest(corpus)
        result = pipeline.query("tell me about it")
    assert result.queries == [
        "solar panel efficiency",
        "photovoltaic electricity",
        "sunlight conversion",
        "Photovoltaic cells in panels turn sunlight into electricity.",
    ]
    assert result.sources[0].endswith("solar.md")


def test_corrective_loop_retries_then_answers(tmp_path, corpus):
    def responder(prompt):
        if "Does this passage" in prompt:
            return "yes" if "Sourdough" in prompt else "no"
        if "did not find useful documents" in prompt:
            return "sourdough starter"
        return answer_from_context(prompt)

    settings = {"top_k": 1, "corrective": {"enabled": True, "max_retries": 2}}
    with make_pipeline(tmp_path, llm=rule_llm(responder), retrieval=settings) as pipeline:
        pipeline.ingest(corpus)
        result = pipeline.query("solar loaf culture")
    assert result.retries == 1
    assert result.queries[-1] == "sourdough starter"
    assert result.sources == [str(corpus / "baking.txt")]
    assert not result.abstained


def test_corrective_loop_abstains_when_nothing_is_relevant(tmp_path, corpus):
    def responder(prompt):
        return "no" if "Does this passage" in prompt else "another query"

    settings = {"corrective": {"enabled": True, "max_retries": 1}}
    llm = rule_llm(responder)
    with make_pipeline(tmp_path, llm=llm, retrieval=settings) as pipeline:
        pipeline.ingest(corpus)
        result = pipeline.query("quantum chromodynamics")
    assert result.abstained and result.documents == []
    assert not any("Context:" in p for p in llm.prompts)


def test_empty_index_abstains_without_calling_llm(tmp_path):
    llm = rule_llm(answer_from_context)
    with make_pipeline(tmp_path, llm=llm) as pipeline:
        result = pipeline.query("anything")
        streamed = "".join(pipeline.stream("anything"))
    assert result.abstained
    assert streamed == pipeline.config.generation.abstain_message
    assert llm.prompts == []


def test_stream_and_async_match_query(tmp_path, corpus):
    with make_pipeline(tmp_path) as pipeline:
        pipeline.ingest(corpus)
        expected = pipeline.query("solar panel efficiency").answer
        assert "".join(pipeline.stream("solar panel efficiency")) == expected
        assert asyncio.run(pipeline.aquery("solar panel efficiency")).answer == expected

        async def collect():
            return "".join([t async for t in pipeline.astream("solar panel efficiency")])

        assert asyncio.run(collect()) == expected


def test_retrieved_text_is_framed_as_data(tmp_path, corpus):
    llm = rule_llm(answer_from_context)
    with make_pipeline(tmp_path, llm=llm) as pipeline:
        pipeline.ingest(corpus)
        pipeline.query("solar panel efficiency")
    assert "not instructions" in llm.prompts[-1]


def test_faiss_with_huggingface_warns_on_macos(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    config = RAGConfig.model_validate(
        {
            "vector_store": {"provider": "faiss"},
            "embeddings": {"provider": "huggingface", "model": "m"},
        }
    )
    with pytest.warns(RuntimeWarning, match="OpenMP"):
        RAGPipeline(config)


def test_missing_components_explain_themselves(tmp_path):
    pipeline = RAGPipeline(RAGConfig(persist_dir=str(tmp_path)))
    with pytest.raises(ConfigurationError, match="No LLM configured"):
        _ = pipeline.llm
    with pytest.raises(ConfigurationError, match="No vector store configured"):
        _ = pipeline.index


def test_delete_source_accepts_relative_paths(tmp_path, corpus, monkeypatch):
    with make_pipeline(tmp_path) as pipeline:
        pipeline.ingest(corpus)
        monkeypatch.chdir(corpus)
        assert pipeline.delete_source("solar.md") == 1
        assert not any(s.endswith("solar.md") for s in pipeline.sources())
