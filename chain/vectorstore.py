"""Vector store access: embedding model + ChromaDB retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PATH = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "chunks.json"
COLLECTION_NAME = "sharan_finance"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match ingest/generate_embeddings.py
DEFAULT_K = 5


@dataclass
class VectorStore:
    """Bundles the persisted ChromaDB collection with the embedding model used to query it."""
    collection: chromadb.Collection
    embedder: SentenceTransformer


def populate_from_chunks(collection: chromadb.Collection, embedder: SentenceTransformer) -> int:
    """Embed every chunk in chunks.json and upsert it into the collection. Returns the chunk count."""
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    ids = [f"{c['video_id']}_{c['start_time']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"video_id": c["video_id"], "title": c["title"], "start_time": c["start_time"], "source_url": c["source_url"]}
        for c in chunks
    ]
    embeddings = embedder.encode(documents).tolist()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)


def build_vectorstore() -> VectorStore:
    """Open the ChromaDB collection, auto-populating it from chunks.json on first run if it's empty.

    data/chroma_db/ isn't committed to git (ChromaDB's binary index files got
    rejected by Hugging Face's push — see git history). So on a fresh clone or
    a fresh Space, this collection starts empty and gets built here instead of
    requiring a separate manual step. Locally, once it's been populated once,
    this is just a fast reopen.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if collection.count() == 0:
        populate_from_chunks(collection, embedder)
    return VectorStore(collection, embedder)


def retrieve(store: VectorStore, question: str, k: int = DEFAULT_K) -> list[dict]:
    """Embed the question and return its top-k nearest transcript chunks (text + metadata)."""
    query_embedding = store.embedder.encode([question]).tolist()
    results = store.collection.query(query_embeddings=query_embedding, n_results=k)
    return [
        {"text": doc, **meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
