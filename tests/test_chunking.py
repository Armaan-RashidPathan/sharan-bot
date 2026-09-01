"""Tests for the chunking logic in ingest/extract_transcripts.py.

These are pure-function tests against synthetic transcript entries — no
network calls, no real YouTube data needed.
"""

from ingest.extract_transcripts import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, chunk_transcript


def make_entries(word_count: int, seconds_per_word: float = 0.4) -> list[dict]:
    """Build synthetic transcript entries, one word per entry, at a fixed cadence.

    Real transcripts group multiple words per entry, but chunk_transcript splits
    entry text on whitespace anyway, so one-word entries exercise the same
    word-by-word logic while keeping start-time math predictable in tests.
    """
    return [
        {"text": f"word{i}", "start": round(i * seconds_per_word, 2)}
        for i in range(word_count)
    ]


def test_short_transcript_produces_one_chunk():
    """Fewer words than CHUNK_SIZE_WORDS should collapse to a single chunk with everything in it."""
    entries = make_entries(50)
    chunks = chunk_transcript(entries, "vid123", "Test Video")

    assert len(chunks) == 1
    assert chunks[0]["text"].split() == [f"word{i}" for i in range(50)]
    assert chunks[0]["start_time"] == 0


def test_empty_transcript_produces_no_chunks():
    assert chunk_transcript([], "vid123", "Test Video") == []


def test_long_transcript_chunk_count_and_overlap():
    """Enough words for several chunks: verify count, size, and that consecutive
    chunks actually overlap by CHUNK_OVERLAP_WORDS words (the whole point of the
    overlap — no answer should get cleanly severed at a chunk boundary)."""
    total_words = CHUNK_SIZE_WORDS * 3  # comfortably more than one chunk's worth
    entries = make_entries(total_words)
    chunks = chunk_transcript(entries, "vid123", "Test Video")

    assert len(chunks) > 1

    # Every chunk except possibly the last is exactly CHUNK_SIZE_WORDS words.
    for chunk in chunks[:-1]:
        assert len(chunk["text"].split()) == CHUNK_SIZE_WORDS

    # Consecutive chunks share exactly CHUNK_OVERLAP_WORDS words at the boundary.
    for prev, curr in zip(chunks, chunks[1:]):
        prev_words = prev["text"].split()
        curr_words = curr["text"].split()
        assert prev_words[-CHUNK_OVERLAP_WORDS:] == curr_words[:CHUNK_OVERLAP_WORDS]

    # No word from the source transcript should be missing from every chunk.
    all_chunked_words = set()
    for chunk in chunks:
        all_chunked_words.update(chunk["text"].split())
    assert all_chunked_words == {f"word{i}" for i in range(total_words)}


def test_chunk_metadata_fields():
    entries = make_entries(20)
    chunks = chunk_transcript(entries, "abc123", "My Video Title")

    chunk = chunks[0]
    assert chunk["video_id"] == "abc123"
    assert chunk["title"] == "My Video Title"
    assert chunk["source_url"] == "https://youtube.com/watch?v=abc123&t=0"
    assert isinstance(chunk["start_time"], int)


def test_start_time_matches_first_word_of_chunk():
    """Each chunk's start_time should be the timestamp of its own first word,
    not the transcript's overall start — this is what makes citation links
    jump to the right moment instead of always the video's beginning."""
    entries = make_entries(CHUNK_SIZE_WORDS * 2)
    chunks = chunk_transcript(entries, "vid123", "Test Video")

    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    for i, chunk in enumerate(chunks):
        expected_word_index = i * step
        expected_start = entries[expected_word_index]["start"]
        assert chunk["start_time"] == int(expected_start)


def test_large_gap_between_entries_forces_a_chunk_boundary():
    """When entries were removed upstream (e.g. strip_crowd_banter), the
    surviving entries on either side of the gap aren't actually adjacent
    speech. A chunk must never span that gap — otherwise two unrelated
    moments in the video get silently merged into one chunk, diluting its
    embedding across two topics instead of representing either well."""
    before_gap = make_entries(30)  # ends at start=11.6s
    after_gap = [
        {"text": f"word{i}", "start": 100.0 + (i - 30) * 0.4}
        for i in range(30, 60)
    ]  # starts at 100.0s — a huge, deliberate gap from the first half

    chunks = chunk_transcript(before_gap + after_gap, "vid123", "Test Video")

    # Both halves are short enough to each be a single chunk on their own,
    # so a gap-aware chunker produces exactly two chunks, not one 60-word chunk.
    assert len(chunks) == 2
    assert chunks[0]["text"].split() == [f"word{i}" for i in range(30)]
    assert chunks[1]["text"].split() == [f"word{i}" for i in range(30, 60)]
    assert chunks[1]["start_time"] == 100


def test_small_gaps_do_not_force_a_boundary():
    """Normal transcript cadence (short gaps between consecutive entries)
    shouldn't trigger segment-splitting — this is the common case and must
    behave exactly like a single continuous stream, as before."""
    entries = make_entries(CHUNK_SIZE_WORDS + 50, seconds_per_word=0.4)
    chunks = chunk_transcript(entries, "vid123", "Test Video")

    assert len(chunks) == 2  # same as the no-gap-awareness behavior
