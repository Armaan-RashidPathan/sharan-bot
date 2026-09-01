# 1% Club Finance Assistant

[![tests](https://github.com/Armaan-RashidPathan/sharan-bot/actions/workflows/tests.yml/badge.svg)](https://github.com/Armaan-RashidPathan/sharan-bot/actions/workflows/tests.yml)

**Live demo: [sharan-bot-hcr.streamlit.app](https://sharan-bot-hcr.streamlit.app/)**

A RAG-based conversational assistant trained exclusively on Sharan Hegde's
(founder of the 1% Club) financial education content — the 291-minute
masterclass plus 12 topic videos from his YouTube channel.

## What it does

- Ask a question about savings, investing, taxes, or insurance
- Answers are grounded only in what Sharan actually teaches — retrieved from
  his transcripts, not the model's generic financial knowledge
- Every answer cites its source video and timestamp, linking straight to
  that moment
- The system prompt is tuned to match Sharan's own teaching style — direct
  address, concrete numbers, rhetorical build-up — not generic advisor tone

## Architecture

```
YouTube transcripts
      ↓  chunking (400 words, 50-word overlap) + metadata tagging
      ↓  embeddings (sentence-transformers: all-MiniLM-L6-v2)
      ↓  ChromaDB (persistent vector store)
      ↓  retrieval (top-5 chunks) + LangChain RAG chain
      ↓  Groq-hosted LLM (openai/gpt-oss-120b)
Streamlit UI ← (FastAPI backend also built + Docker-tested independently;
                the deployed demo calls the chain in-process for simplicity)
```

## Project structure

```
ingest/       transcript extraction, chunking, and vector store population
chain/        retrieval + the RAG chain — the core of the project
backend/      FastAPI API layer (POST /ask) — built and Docker-tested,
              not what serves the live demo (see claude.md for why)
frontend/     Streamlit UI
data/         chunks.json (extracted transcripts, committed);
              chroma_db/ is generated on first run, not committed
tests/        pytest suite for the chunking and citation-formatting logic
evaluation/   retrieval evaluation harness — recall@k / MRR against a
              30-question hand-labeled golden set (see below)
```

## Retrieval evaluation

`claude.md` originally tuned the retriever "by feel" — chunk size, overlap,
and k were all judgment calls with no way to check them. `evaluation/`
replaces that with a measured baseline: 30 real questions, each hand-labeled
against the specific transcript chunk that answers it (picked by reading the
actual corpus, not generated), scored on recall@k and MRR.

```bash
./.venv/Scripts/python.exe -m evaluation.retrieval_eval
```

Current baseline (dense retrieval only, no reranking): **recall@5 = 0.80**,
**recall@10 = 0.93**, **MRR = 0.638**. The two misses are diagnosed by hand,
not just reported as a number — see [`evaluation/RESULTS.md`](evaluation/RESULTS.md)
for what they reveal about where dense-embedding retrieval actually breaks
down (prescriptive vs. descriptive framing of the same topic), and what a
fix would look like.

## Testing

```bash
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pytest tests/ -v
```

Tests cover the deterministic logic — transcript chunking (chunk size,
overlap, timestamp-per-chunk correctness) and the citation-formatting
helpers — as pure-function tests against synthetic input. No API key or
network access needed to run them, which is also why CI can run on every
push without secrets. See `.github/workflows/tests.yml`.

## Running locally

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Create a `.env` file with a [free Groq API key](https://console.groq.com):

```
GROQ_API_KEY=your_key_here
```

Then run the frontend — it builds the vector store from `data/chunks.json`
automatically on first launch:

```bash
./.venv/Scripts/python.exe -m streamlit run frontend/app.py
```

## Author

**Armaan Rashid Pathan** — [GitHub](https://github.com/Armaan-RashidPathan)

See [`claude.md`](claude.md) for the full architecture notes and design
decisions behind this project.
