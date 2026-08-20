"""
Layer 4: FastAPI backend exposing the Sharan Hegde finance RAG assistant over HTTP.

Sits between the retrieval-augmented QA chain (chain/qa_chain.py) and whatever
calls it — the Streamlit frontend planned for Layer 5, or manual testing via
curl/the /docs Swagger UI. Importing chain.qa_chain at module load time builds
the vector store connection and the LLM client once at server startup; every
request then reuses those same objects rather than rebuilding them.

Endpoints:
    GET  /health  - liveness check
    POST /ask     - ask a question, get back a grounded answer + source citations

Run from the project root:
    ./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import AskRequest, AskResponse
from chain.qa_chain import ask as run_qa_chain

app = FastAPI(
    title="1% Club Finance Assistant API",
    description="RAG-based Q&A over Sharan Hegde's financial education content.",
    version="0.1.0",
)

# The Streamlit frontend (Layer 5) will run on a different local port than this
# API during development, so browser requests need CORS explicitly allowed.
# Left permissive ("*") since this is a single-purpose demo, not a
# multi-tenant service handling untrusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness check — deployment platforms (e.g. Hugging Face Spaces) poll this to confirm the service is up."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run the RAG chain on the submitted question and return the answer with its source citations."""
    result = run_qa_chain(request.question)
    return AskResponse(answer=result["answer"], sources=result["sources"])
