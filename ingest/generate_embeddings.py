"""
Layer 2: Manually rebuild the vector store from scratch.

This wipes and repopulates the collection unconditionally — use it after
re-chunking (e.g. changed chunk size in extract_transcripts.py) to force a
full re-embed. chain.vectorstore.build_vectorstore() does the same
population logic automatically but only when the collection is empty (first
run on a fresh clone/Space); this script is for deliberately starting over.
"""

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from chain.vectorstore import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, populate_from_chunks


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass  # collection didn't exist yet — nothing to delete

    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    count = populate_from_chunks(collection, embedder)
    print(f"Added {count} chunks to '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
