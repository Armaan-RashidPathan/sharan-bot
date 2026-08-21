# 1% Club Finance Assistant

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
ingest/     transcript extraction, chunking, and vector store population
chain/      retrieval + the RAG chain — the core of the project
backend/    FastAPI API layer (POST /ask) — built and Docker-tested,
            not what serves the live demo (see claude.md for why)
frontend/   Streamlit UI
data/       chunks.json (extracted transcripts, committed);
            chroma_db/ is generated on first run, not committed
```

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
