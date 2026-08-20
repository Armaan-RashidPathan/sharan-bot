# 1% Club Financial AI Assistant — Project Context for Codex

## What This Project Is

A RAG (Retrieval-Augmented Generation) based conversational AI assistant trained
exclusively on Sharan Hegde's financial education content. Sharan Hegde is the
founder of the 1% Club — a fintech/financial literacy platform in India targeting
young working professionals.

The purpose of this project is twofold:
1. Build a genuinely useful AI tool for the 1% Club's audience
2. Use it as a demonstration project to reach out to Sharan Hegde directly for
   an internship or role, showing the ability to build AI products — which he has
   publicly stated is a hiring priority

---

## Source Material

| Source | Type | Notes |
|---|---|---|
| "Ultimate Finance Course You Need To Watch in Your 20s" (5-hour video) | YouTube seminar | Primary source — Sharan teaching directly with crowd Q&A |
| 14-video playlist (~15 min each, ~3.5 hrs total) | YouTube playlist | Secondary source — topic-specific videos |
| Blog posts / LinkedIn articles (if found) | Written content | Add if available |

**Do NOT use:**
- Podcast episodes where Sharan is a guest (too conversational, low signal density)
- Videos where Sharan is interviewing someone else

**YouTube Video IDs (finalized — see `ingest/extract_transcripts.py` for the full
list with titles):**
```python
VIDEO_IDS = {
    "5hr_seminar": "IgjhqPgwwGI",  # "Everything You Need to Know About Money in Your 20s | 291-Minute Masterclass | Ft. Sharan Hegde"
    "playlist": [
        "GjfjqfqDzCg", "qsibNjO2Cvc", "Iq_AftOxqMY", "DET3gmXW1uY",
        "eNmWnAl0U80", "YiHop4ooqKQ", "RoiVyNe_qy8", "Ab4Kzz3kWUk",
        "uUdM-kiCOvA", "CnM5E-frf8s", "DcNToAAISc4", "Y3cKly68pWI",
        # 12 of the 14 playlist videos — 2 excluded as guest interview / panel content:
        # ngY_m_9gwCk (Monika Halan interview), EsMGwG_H2qA (Budget 2025 expert panel)
    ]
}
```

---

## Architecture Overview

```
YouTube Videos
      ↓
Transcript Extraction (youtube-transcript-api)
      ↓
Chunking + Metadata Tagging (400 words, 50-word overlap)
      ↓
Embedding Generation (sentence-transformers: all-MiniLM-L6-v2)
      ↓
Vector Store — ChromaDB (persistent, local)
      ↓
LangChain RetrievalQA Chain (k=5 chunks retrieved)
      ↓
LLM — GPT-3.5-turbo (temperature=0.2)
      ↓
FastAPI Backend (/ask endpoint)
      ↓
Streamlit Frontend (with example questions + source citations)
      ↓
Deploy — Hugging Face Spaces (free, gives live shareable URL)
```

---


---

## Layer 1 — Transcript Extraction

**File:** `ingest/extract_transcripts.py`

Key decisions:
- Chunk size: **400 words** (sweet spot for seminar content — preserves context)
- Overlap: **50 words** between chunks (prevents answers being cut at boundaries)
- Each chunk carries metadata: video_id, title, start_time, source_url with timestamp
- The source_url format: `https://youtube.com/watch?v={video_id}&t={start_time}`
  so citations link directly to the exact moment in the video

```python

---

## Three Things That Make or Break Quality

1. **Chunk size** — 400 words with 50-word overlap. Too small = lost context.
   Too large = imprecise retrieval. Do not change this without testing.

2. **System prompt** — this is what makes the assistant sound grounded in
   Sharan's framework vs. giving generic financial advice. Spend time tuning it.
   Test with at least 15-20 real questions before shipping.

3. **k=5 retriever setting** — retrieves top 5 chunks per query. If answers
   feel shallow, increase to 7. If unfocused, drop to 3.

---

## Why This Project Exists — Outreach Context

This is being built as a demonstration project to reach out directly to
Sharan Hegde (founder, 1% Club) for an internship or role.

Sharan has publicly stated he wants to hire people who are experts at building
AI products. The strategy is:
- Build a working, deployed demo (NOT just a GitHub repo)
- Reach out via LinkedIn with the live URL
- Keep the message under 150 words
- Ask only for a 15-minute conversation, not a job directly

The outreach message will be drafted separately once the demo is live and
has a shareable Hugging Face Spaces URL.

**The single most important rule: do not reach out until the demo is live
and clickable. A working link beats any resume or cover letter.**

---

## About the Developer

**Armaan Rashid Pathan**
- 3rd year B.Tech CSE (AI & ML specialization), VIT Chennai, CGPA 8.64
- AI Developer Intern at Unfaro AI — built production multi-agent LLM pipelines
  (LangChain, FastAPI, Python), shipped backend features on a live B2B SaaS platform
- Built a FinBERT/VADER sentiment-driven RL trading agent (PPO, Gymnasium)
  with real held-out test results (Sharpe 1.57 vs 1.23 baseline)
- Stack: Python, FastAPI, LangChain, LangGraph, PostgreSQL, AWS, PyTorch
- GitHub: github.com/Armaan-RashidPathan

This background makes the 1% Club demo directly relevant — financial NLP
(FinBERT) and multi-agent pipeline experience are exactly what's needed here.
```