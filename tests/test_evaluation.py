import json
import math

import pytest
from langchain_core.documents import Document

from ragfoundry import EvalSample, load_dataset
from ragfoundry.evaluation import (
    answer_correctness,
    average_precision,
    faithfulness,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragfoundry.evaluation.runner import retrieval_keys

from .conftest import rule_llm
from .test_pipeline import make_pipeline


def test_rank_metrics_against_hand_computed_values():
    retrieved = ["x", "a", "y", "b"]
    relevant = {"a", "b", "c"}
    assert hit_rate(retrieved, relevant, 1) == 0
    assert hit_rate(retrieved, relevant, 2) == 1
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert mrr(retrieved, relevant) == 0.5
    assert average_precision(retrieved, relevant) == pytest.approx((1 / 2 + 2 / 4) / 3)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(retrieved, relevant, 4) == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(5)) / ideal
    )


def test_repeated_relevant_item_counts_once_for_recall_and_ndcg():
    retrieved = ["a", "a"]
    assert precision_at_k(retrieved, {"a"}, 2) == 1
    assert recall_at_k(retrieved, {"a"}, 2) == 1
    assert ndcg_at_k(retrieved, {"a"}, 2) == 1
    assert average_precision(retrieved, {"a"}) == 1


def test_labels_by_source_suffix_and_reference_passage():
    documents = [
        Document(page_content="Unrelated.", metadata={"source": "/data/x.txt"}),
        Document(page_content="The Pump  needs a gasket.", metadata={"source": "/data/manual.pdf"}),
    ]
    sample = EvalSample(
        question="q", relevant_sources=["manual.pdf"], reference_contexts=["pump needs a gasket"]
    )
    keys, gold = retrieval_keys(documents, sample)
    assert gold == {"source:manual.pdf", "context:0"}
    assert keys[0].startswith("miss") and keys[1] in gold


def test_faithfulness_counts_supported_claims():
    def judge(prompt):
        if "List the distinct factual claims" in prompt:
            return '["Paris is in France.", "Paris has 90 million residents."]'
        return "yes" if "Claim: Paris is in France." in prompt else "no"

    llm = rule_llm(judge)
    assert faithfulness(llm, "answer", ["Paris is the capital of France."]) == 0.5
    assert faithfulness(llm, "answer", []) is None


def test_claims_given_as_objects_are_accepted():
    def judge(prompt):
        if "List the distinct factual claims" in prompt:
            return '[{"claim": "Cells are 20 percent efficient."}]'
        return "yes"

    assert faithfulness(rule_llm(judge), "About 20 percent.", ["ctx"], question="q") == 1.0


def test_rubric_scores_are_clamped_and_unparseable_is_none():
    assert answer_correctness(rule_llm(lambda p: '{"score": 1.7}'), "q", "a", "r") == 1.0
    assert answer_correctness(rule_llm(lambda p: "great answer"), "q", "a", "r") is None


def test_pipeline_evaluation_report(tmp_path, corpus):
    def responder(prompt):
        if "List the distinct factual claims" in prompt:
            return '["claim"]'
        if "directly supported" in prompt or "Does this passage" in prompt:
            return "yes"
        if '"score"' in prompt:
            return '{"score": 0.8}'
        return "Solar panels are about 20 percent efficient [1]."

    dataset = tmp_path / "eval.jsonl"
    rows = [
        {
            "question": "solar panel efficiency",
            "relevant_sources": ["solar.md"],
            "reference_answer": "About 20 percent.",
        },
        {"question": "sourdough starter feeding", "relevant_sources": ["baking.txt"]},
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_dataset(dataset)) == 2

    with make_pipeline(tmp_path, llm=rule_llm(responder)) as pipeline:
        pipeline.ingest(corpus)
        report = pipeline.evaluate(dataset, k_values=[1, 3])
        retrieval_only = pipeline.evaluate(dataset, mode="retrieval", k_values=[1])

    assert report.summary["hit_rate@1"] == {"mean": 1.0, "count": 2}
    assert report.summary["faithfulness"]["mean"] == 1.0
    assert report.summary["answer_correctness"] == {"mean": 0.8, "count": 1}
    assert "faithfulness" not in retrieval_only.summary
    out = tmp_path / "report.json"
    report.save(out)
    assert json.loads(out.read_text())["rows"][0]["answer"].startswith("Solar")


def test_failing_metric_is_recorded_not_fatal(tmp_path, corpus):
    def responder(prompt):
        if '"score"' in prompt:
            raise RuntimeError("judge unavailable")
        return "yes" if "Does this passage" in prompt else '["claim"]'

    with make_pipeline(tmp_path, llm=rule_llm(responder)) as pipeline:
        pipeline.ingest(corpus)
        report = pipeline.evaluate([EvalSample(question="solar panels")], mode="generation")
    assert report.summary["answer_relevance"]["count"] == 0
    assert "judge unavailable" in report.rows[0]["errors"][0]
