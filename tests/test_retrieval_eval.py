"""Tests for the recall@k / MRR math in evaluation/retrieval_eval.py itself —
against a fake retriever with a known, controlled ranking, not the real
vector store, so this stays fast and deterministic."""

from unittest.mock import patch

import pytest

from evaluation.retrieval_eval import chunk_id, evaluate, resolve_golden_set


def fake_chunk(video_id: str, start_time: int, text: str = "x") -> dict:
    return {"video_id": video_id, "start_time": start_time, "title": "t", "text": text}


def test_chunk_id_format():
    assert chunk_id(fake_chunk("abc123", 90)) == "abc123_90"


def test_resolve_golden_set_finds_chunk_containing_fingerprint():
    chunks = [
        fake_chunk("v", 10, text="talking about SIPs and how much to invest monthly"),
        fake_chunk("v", 20, text="something unrelated about electricity bills"),
    ]
    golden_set = [{"question": "what are SIPs?", "answer_fingerprint": "how much to invest monthly"}]

    resolved, unresolved = resolve_golden_set(golden_set, chunks)

    assert unresolved == []
    assert resolved == [{"question": "what are SIPs?", "relevant_ids": ["v_10"]}]


def test_resolve_golden_set_reports_unresolved_fingerprints_separately():
    """A fingerprint missing from the corpus (e.g. stripped by upstream
    cleaning) is a distinct failure mode from a retrieval miss — it must be
    reported separately, not silently scored as rank=None."""
    chunks = [fake_chunk("v", 10, text="something else entirely")]
    golden_set = [{"question": "q1", "answer_fingerprint": "text that does not exist anywhere"}]

    resolved, unresolved = resolve_golden_set(golden_set, chunks)

    assert resolved == []
    assert unresolved == golden_set


def test_resolve_golden_set_can_match_multiple_overlapping_chunks():
    """A fingerprint sitting in an overlap region can legitimately appear in
    more than one chunk — resolve_golden_set should keep every match, not
    just the first."""
    chunks = [
        fake_chunk("v", 10, text="shared phrase across the overlap boundary"),
        fake_chunk("v", 20, text="shared phrase across the overlap boundary too"),
    ]
    golden_set = [{"question": "q1", "answer_fingerprint": "shared phrase across the overlap boundary"}]

    resolved, unresolved = resolve_golden_set(golden_set, chunks)

    assert unresolved == []
    assert resolved[0]["relevant_ids"] == ["v_10", "v_20"]


def test_evaluate_recall_and_mrr_with_known_ranking():
    # Three questions, each looking for chunk "v_1", planted at a controlled
    # rank in a fake retriever's results: rank 1, rank 3, and never present.
    golden_set = [
        {"question": "q1", "relevant_ids": ["v_1"]},
        {"question": "q2", "relevant_ids": ["v_1"]},
        {"question": "q3", "relevant_ids": ["v_1"]},
    ]

    canned_results = {
        "q1": [fake_chunk("v", 1), fake_chunk("v", 2), fake_chunk("v", 3)],  # rank 1
        "q2": [fake_chunk("v", 2), fake_chunk("v", 3), fake_chunk("v", 1)],  # rank 3
        "q3": [fake_chunk("v", 2), fake_chunk("v", 3), fake_chunk("v", 4)],  # not found
    }

    def fake_retrieve(store, question, k):
        return canned_results[question]

    with patch("evaluation.retrieval_eval.retrieve", side_effect=fake_retrieve):
        results = evaluate(store=None, golden_set=golden_set)

    ranks = {q["question"]: q["rank"] for q in results["per_question"]}
    assert ranks == {"q1": 1, "q2": 3, "q3": None}

    # recall@3: q1 (rank 1) and q2 (rank 3) count, q3 never found -> 2/3.
    # Ranks never exceed 3 here, so recall is identical for every larger k.
    for k, recall in results["recall_at_k"].items():
        assert recall == pytest.approx(2 / 3), f"recall@{k} should be 2/3"

    # MRR = mean(1/1, 1/3, 0)
    assert results["mrr"] == pytest.approx((1 / 1 + 1 / 3 + 0) / 3)
