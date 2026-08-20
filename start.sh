#!/bin/bash
# Launches both services inside the single container Hugging Face Spaces runs.
#
# The FastAPI backend is only ever called by the Streamlit frontend running in
# the same container, so it's bound to 127.0.0.1 rather than 0.0.0.0 — it's
# never reachable from outside the container, which avoids exposing an
# unauthenticated endpoint that could burn through the GROQ_API_KEY's free-tier
# rate limit if it were public. Only Streamlit, on 7860, is exposed (Spaces'
# Docker SDK expects the app to listen on that port).
set -e

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# Wait for the backend to actually answer /health before starting Streamlit,
# rather than guessing a fixed delay. On a fresh container (no cached model
# weights, unlike a local .venv that's already downloaded them once) loading
# the embedding model and opening the ChromaDB collection can take well over
# 30 seconds — a fixed short sleep here left Streamlit sending its first
# requests to a backend that wasn't listening yet. No curl in this slim base
# image, so the check is a small inline Python urlopen call. Capped at 2
# minutes so a genuinely broken backend fails fast and visibly in the
# container logs instead of hanging forever.
echo "Waiting for backend to become ready..."
for i in $(seq 1 60); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then
        echo "Backend is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Backend did not become ready within 2 minutes." >&2
        exit 1
    fi
    sleep 2
done

streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
