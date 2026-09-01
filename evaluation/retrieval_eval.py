"""
Retrieval evaluation harness.

Measures how well chain.vectorstore.retrieve() finds the right transcript
chunk for a question, against a hand-labeled golden set (golden_set.json):
30 real questions, each paired with a verbatim "answer_fingerprint" phrase —
picked by reading the actual transcript content across all 13 source videos
— rather than a hardcoded chunk ID.

Why a fingerprint instead of a chunk ID: chunking runs on a flattened,
continuous word stream, so a chunk's start_time (and therefore its ID)
shifts the moment anything upstream changes — re-chunking with a different
size, or stripping noisy transcript segments. A hardcoded ID silently goes
stale in that case, quietly invalidating the whole eval. A verbatim phrase
gets re-resolved against whatever chunks.json currently contains, so the
golden set survives reprocessing the pipeline.

Metrics:
- Recall@k: fraction of questions where a known-relevant chunk appears
  somewhere in the top-k retrieved results.
- MRR (Mean Reciprocal Rank): average of 1/rank of the first relevant chunk
  found (0 if it isn't found within the largest k evaluated).

Known limitation: each question is labeled against the ONE chunk it was
written from, not every chunk that could legitimately answer it. A retrieval
that finds a different, equally valid chunk (the same topic discussed
elsewhere in the corpus, or an adjacent overlapping chunk) still counts as a
miss here. So these numbers are a conservative lower bound, useful for
comparing retrieval configurations against each other, not an absolute
correctness score.

Usage (from the project root):
    ./.venv/Scripts/python.exe -m evaluation.retrieval_eval
"""

import json
from pathlib import Path

from chain.vectorstore import CHUNKS_PATH, build_vectorstore, retrieve

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
K_VALUES = [3, 5, 7, 10]
MAX_K = max(K_VALUES)


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def chunk_id(chunk: dict) -> str:
    return f"{chunk['video_id']}_{chunk['start_time']}"


def resolve_golden_set(golden_set: list[dict], chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Resolve each entry's answer_fingerprint to whichever chunk(s) in the
    current corpus contain that verbatim phrase.

    Returns (resolved, unresolved). An unresolved entry (its fingerprint
    isn't found in any current chunk) is a distinct failure mode from a
    retrieval miss — it means the underlying content moved, got rephrased,
    or was stripped out upstream (e.g. by transcript cleaning), not that
    retrieval failed to find something that's there.
    """
    resolved, unresolved = [], []
    for item in golden_set:
        fp = item["answer_fingerprint"]
        matches = [chunk_id(c) for c in chunks if fp in c["text"]]
        if matches:
            resolved.append({"question": item["question"], "relevant_ids": matches})
        else:
            unresolved.append(item)
    return resolved, unresolved


def evaluate(store, golden_set: list[dict]) -> dict:
    """Retrieve once per question at MAX_K, then derive every k's recall by
    truncating that same ranked list — avoids re-querying per k value."""
    per_question = []
    for item in golden_set:
        retrieved = retrieve(store, item["question"], k=MAX_K)
        retrieved_ids = [chunk_id(c) for c in retrieved]
        relevant_ids = set(item["relevant_ids"])

        rank = next(
            (i for i, rid in enumerate(retrieved_ids, start=1) if rid in relevant_ids),
            None,
        )

        per_question.append({
            "question": item["question"],
            "relevant_ids": item["relevant_ids"],
            "retrieved_ids": retrieved_ids,
            "rank": rank,
        })

    n = len(golden_set)
    by_k = {
        k: sum(1 for q in per_question if q["rank"] is not None and q["rank"] <= k) / n
        for k in K_VALUES
    }
    mrr = sum((1 / q["rank"]) if q["rank"] else 0 for q in per_question) / n

    return {"per_question": per_question, "recall_at_k": by_k, "mrr": mrr}


def print_report(results: dict, unresolved: list[dict]) -> None:
    if unresolved:
        print(f"WARNING: {len(unresolved)} golden-set fingerprint(s) not found in the current corpus:")
        for item in unresolved:
            print(f"  - {item['question']!r} (fingerprint: {item['answer_fingerprint']!r})")
        print()

    n = len(results["per_question"])
    print(f"Questions evaluated: {n}\n")
    print(f"{'k':>4} | recall@k")
    print("-" * 20)
    for k, recall in results["recall_at_k"].items():
        print(f"{k:>4} | {recall:.2f}")
    print(f"\nMRR: {results['mrr']:.3f}")

    misses = [q for q in results["per_question"] if q["rank"] is None or q["rank"] > 5]
    if misses:
        print(f"\nQuestions where the labeled chunk missed the top 5 ({len(misses)}):")
        for q in misses:
            rank_label = q["rank"] if q["rank"] else "not found"
            print(f"  - [{rank_label}] {q['question']}")


def main():
    store = build_vectorstore()
    chunks = load_chunks()
    raw_golden_set = load_golden_set()
    golden_set, unresolved = resolve_golden_set(raw_golden_set, chunks)

    results = evaluate(store, golden_set)
    print_report(results, unresolved)

    results["unresolved"] = unresolved
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull per-question results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
