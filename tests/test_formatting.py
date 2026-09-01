"""Tests for chain/formatting.py — the pure helpers that shape retrieved
chunks into prompt context and into client-facing citations."""

from chain.formatting import format_context, to_citations

SAMPLE_CHUNKS = [
    {
        "video_id": "abc123",
        "title": "Video One",
        "start_time": 90,
        "source_url": "https://youtube.com/watch?v=abc123&t=90",
        "text": "First chunk of transcript text.",
    },
    {
        "video_id": "def456",
        "title": "Video Two",
        "start_time": 15,
        "source_url": "https://youtube.com/watch?v=def456&t=15",
        "text": "Second chunk of transcript text.",
    },
]


def test_format_context_includes_source_and_text_for_every_chunk():
    result = format_context(SAMPLE_CHUNKS)
    for chunk in SAMPLE_CHUNKS:
        assert chunk["title"] in result
        assert str(chunk["start_time"]) in result
        assert chunk["text"] in result


def test_format_context_empty_list():
    assert format_context([]) == ""


def test_to_citations_keeps_only_client_facing_fields():
    citations = to_citations(SAMPLE_CHUNKS)

    assert len(citations) == len(SAMPLE_CHUNKS)
    for citation, chunk in zip(citations, SAMPLE_CHUNKS):
        assert citation == {
            "title": chunk["title"],
            "start_time": chunk["start_time"],
            "source_url": chunk["source_url"],
        }
        # The raw transcript text and video_id shouldn't leak into the citation.
        assert "text" not in citation
        assert "video_id" not in citation


def test_to_citations_empty_list():
    assert to_citations([]) == []
