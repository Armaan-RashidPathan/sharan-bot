# Retrieval Evaluation

Run: `./.venv/Scripts/python.exe -m evaluation.retrieval_eval`

This documents three iterations against the same 30-question golden set:
a baseline, a first attempt at fixing a problem the baseline exposed, and a
correction after that first attempt revealed a second, deeper problem. The
numbers didn't end up cleanly better — that's reported honestly below,
along with why.

## Iteration 1 — Baseline (dense retrieval, no cleaning)

| k  | recall@k |
|----|----------|
| 3  | 0.73     |
| 5  | 0.80     |
| 7  | 0.83     |
| 10 | 0.93     |

**MRR: 0.638**

Two misses were diagnosed by hand rather than left as a bare number. One —
*"What's the typical asset allocation breakdown for the average Indian
household?"* — retrieved Sharan's own *recommended* allocation instead of
the *actual average household* data the question was labeled against: a
real dense-embedding weakness (prescriptive vs. descriptive framing of the
same topic). Digging into the corpus for that also surfaced a bigger,
separate problem: the seminar transcript contains crowd Q&A crosstalk
(marked with a literal `>>` per speaker turn) sitting in the same chunks as
Sharan's actual teaching, burning retrieval slots on near-noise.

## Iteration 2 — Strip crowd banter (naive: no chunk-boundary awareness)

`ingest/extract_transcripts.py`'s `strip_crowd_banter` detects and removes
transcript entries inside a crowd-Q&A stretch (sliding-window density of
`>>` markers). Result: **1,791 of 6,952 entries (25.8%) removed from the
seminar** — the other 12 solo-monologue videos lost nothing, confirming
the detector doesn't false-positive on clean content.

Re-running the eval surfaced something the fingerprint-matching methodology
(see below) caught automatically: **2 of 30 golden-set fingerprints no
longer resolved to any chunk.** Investigating both by hand:

- One (*"What counts as passive income?"*) was **correctly removed** — the
  original chunk I'd labeled it against turned out to itself be crowd Q&A
  (Sharan asking the audience for examples, not him teaching). The golden
  label was flawed from the start; the cleaning fixed my own mistake.
- The other (*HDFC reward points*) had its content survive, just with
  slightly different exact wording at the removal boundary — a real
  limitation of verbatim-substring fingerprint matching, not of the
  cleaning itself.

More importantly, **two previously-fine questions newly missed the top 10**
(the "free credit period" and "house value" questions). Diagnosis: chunking
treats surviving entries as one continuous stream, so removing a stretch of
entries silently merges the unrelated content on either side of that gap
into a single chunk — diluting its embedding across two topics instead of
representing either one well. This is a chunking bug, not a cleaning
problem: fixed-size chunking was never gap-aware to begin with, and
`strip_crowd_banter` just made the gaps large enough to matter.

## Iteration 3 — Gap-aware chunking

Fixed the actual bug: `chunk_transcript` now splits into independent
segments wherever the gap between consecutive entries exceeds
`MAX_ENTRY_GAP_SECONDS`, so a chunk can never span content that wasn't
really adjacent. The threshold (45s) was picked by sweeping values against
this corpus and measuring what fraction of resulting chunks end up under
100 words (too little context to be useful) — 5s fragmented 52% of chunks
by tripping on ordinary speech pauses (measured max ~12.6s in a clean
video); 45s drops that to ~7% while still separating genuine removed-content
gaps (which run into the tens-to-hundreds of seconds) from normal cadence.

This **did fix the specific mechanism diagnosed above**: both "house value"
(not-found → rank 1) and "asset allocation" (not-found → rank 3) recovered
once their content stopped being merged with unrelated adjacent material.

| k  | recall@k |
|----|----------|
| 3  | 0.71     |
| 5  | 0.79     |
| 7  | 0.82     |
| 10 | 0.86     |

**MRR: 0.596** (n=28 — the 2 unresolved fingerprints from iteration 2 stay unresolved)

## The honest bottom line

Fixing the diagnosed cases didn't produce a net win on the golden set —
recall@5 and MRR are both slightly *below* the original baseline. Why: a
handful of *other* questions ("three bosses", "SIP protection", "free
credit period on its own") now miss for the **same underlying reason** that
gap-awareness didn't (and can't) fix — the answer sits 60-70% of the way
through a 400-word chunk that covers a different subtopic first. Sharan's
speaking style naturally flows between related ideas inside any ~2-3 minute
span; fixed-size chunking doesn't know where one subtopic ends and another
begins, so content deep in a multi-topic chunk gets diluted in that
chunk's embedding regardless of where the chunk's boundaries happen to
fall.

**Kept both changes anyway.** Gap-aware chunking is correct independent of
whether it moves this particular metric — a chunker silently merging
non-adjacent content is a bug regardless of net eval score — and the
corpus is now genuinely free of a real correctness problem: crowd banter
can no longer surface as a cited "source" in the deployed app, something
the small-sample recall number doesn't capture at all.

**What this points to as the real next step:** not more threshold tuning,
but **semantic chunking** — splitting on topic shifts (e.g. embedding
similarity drops between consecutive sentences) instead of a fixed word
count, so a chunk's content is more likely to be about one thing. That's
the fix the diagnosis above actually argues for.

## A known limitation of this methodology

Each question is labeled against the one chunk it was originally written
from (via a verbatim `answer_fingerprint`, resolved fresh against whatever
`chunks.json` currently contains — not a hardcoded chunk ID, since those go
stale the moment chunking changes upstream). A retrieval that surfaces a
different, equally valid chunk — the same topic covered elsewhere in the
corpus, or a neighboring overlapping chunk — still scores as a miss here.
So these numbers are a conservative lower bound, useful for comparing
configurations against each other, not a claim about absolute answer
quality.
