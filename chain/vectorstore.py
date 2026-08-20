"""Vector store access: embedding model + ChromaDB retrieval."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "sharan_finance"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match ingest/generate_embeddings.py
DEFAULT_K = 5


@dataclass
class VectorStore:
    """Bundles the persisted ChromaDB collection with the embedding model used to query it."""
    collection: chromadb.Collection
    embedder: SentenceTransformer


def build_vectorstore() -> VectorStore:
    """Open the ChromaDB collection written by ingest/generate_embeddings.py and load the embedder."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return VectorStore(collection, embedder)


def retrieve(store: VectorStore, question: str, k: int = DEFAULT_K) -> list[dict]:
    """Embed the question and return its top-k nearest transcript chunks (text + metadata)."""
    query_embedding = store.embedder.encode([question]).tolist()
    results = store.collection.query(query_embeddings=query_embedding, n_results=k)
    return [
        {"text": doc, **meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
