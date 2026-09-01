"""
Pure, dependency-free formatting helpers used by the RAG chain.

Split out from qa_chain.py so they're testable without pulling in the LLM
client, the embedding model, or ChromaDB — this module imports nothing but
the standard library.
"""


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into the block of text that fills {context} in the prompt."""
    return "\n\n".join(
        f"[Source: {c['title']} @ {c['start_time']}s]\n{c['text']}" for c in chunks
    )


def to_citations(chunks: list[dict]) -> list[dict]:
    """Strip retrieved chunks down to just the fields a client needs to show a citation."""
    return [
        {"title": c["title"], "start_time": c["start_time"], "source_url": c["source_url"]}
        for c in chunks
    ]
