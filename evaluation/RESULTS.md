# Retrieval Evaluation — Baseline

Run: `./.venv/Scripts/python.exe -m evaluation.retrieval_eval`
Config at time of this run: dense retrieval only (cosine similarity over
`all-MiniLM-L6-v2` embeddings), no reranking, 30-question golden set.

| k  | recall@k |
|----|----------|
| 3  | 0.73     |
| 5  | 0.80     |
| 7  | 0.83     |
| 10 | 0.93     |

**MRR: 0.638**

`k=5` (the current production setting) finds the labeled chunk in 80% of
cases. Jumping to `k=10` closes most of the remaining gap (0.93) — a
data-backed reason to reconsider the k=5/k=7 tradeoff originally
called by feel ("if answers feel shallow, increase to 7"), rather than by
measurement.

## What the misses actually look like

Two questions didn't find their labeled chunk anywhere in the top 10. Both
diagnosed by hand rather than left as an unexplained number:

**"What's the typical asset allocation breakdown for the average Indian
household?"** — retrieval's top hit was Sharan's *recommended* allocation
(60% domestic equity / 10% US equity / 15% debt / ...), not the *actual
average household* breakdown (FDs, LIC, provident fund, cash) the question
was labeled against. Both chunks are legitimately "about asset allocation";
dense cosine similarity doesn't reliably distinguish *prescriptive* framing
("here's what you should do") from *descriptive* framing ("here's what
people actually do") when the vocabulary overlaps this heavily. A likely
fix: hybrid retrieval (BM25 + dense) or a reranking pass, since keyword
overlap alone might separate "should" framing from "is" framing better than
embedding similarity does here.

**"How much tax can I save through investments made for my child's
education?"** — missed the exact labeled chunk, but the top-2 results were
both *adjacent* chunks from the same video, also genuinely about family tax
deductions. This is evidence for, not against, the retrieval quality: it
found substantively relevant content, just not the one specific chunk this
question happened to be authored from.

## A known limitation of this methodology

Each question is labeled with the single chunk it was written from, not
every chunk that could legitimately answer it. A retrieval that surfaces a
different, equally valid chunk — the same topic covered elsewhere in the
corpus, or a neighboring overlapping chunk — still scores as a miss here.
So these numbers are a conservative lower bound, useful for comparing
retrieval configurations against each other (before/after a change), not a
claim about absolute answer quality.

## Next steps this baseline enables

- Try hybrid retrieval (BM25 + dense) or a cross-encoder reranking pass,
  re-run this same harness, and compare recall@k / MRR directly against
  this baseline — turning "I think this helped" into a number.
- The prescriptive-vs-descriptive miss above suggests reranking may help
  more than raising k alone, since raising k just gives the LLM more
  (possibly still-wrong-framing) context rather than fixing the ranking.
