"""Layer 2: Generate and persist embeddings for transcript chunks."""

import json
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "sharan_finance"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    ids = [f"{chunk['video_id']}_{chunk['start_time']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "video_id": chunk["video_id"],
            "title": chunk["title"],
            "start_time": chunk["start_time"],
            "source_url": chunk["source_url"],
        }
        for chunk in chunks
    ]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass  # collection didn't exist yet — nothing to delete
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Added {len(chunks)} chunks to '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
