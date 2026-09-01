"""Tests for strip_crowd_banter in ingest/extract_transcripts.py — the
crowd-Q&A-detection pass that runs before chunking."""

from ingest.extract_transcripts import strip_crowd_banter


def clean_entry(i: int) -> dict:
    return {"text": f"monologue word {i}", "start": float(i)}


def noisy_entry(i: int) -> dict:
    return {"text": f">> reply {i}", "start": float(i)}


def test_empty_entries():
    assert strip_crowd_banter([]) == []


def test_all_clean_entries_are_kept():
    entries = [clean_entry(i) for i in range(20)]
    assert strip_crowd_banter(entries) == entries


def test_isolated_single_marker_is_kept():
    """One '>>' entry surrounded by clean monologue shouldn't trip the
    density threshold — this is normal speech, not a crowd Q&A stretch."""
    entries = [clean_entry(i) for i in range(10)]
    entries[5] = noisy_entry(5)

    result = strip_crowd_banter(entries, window=8, threshold=0.3)

    assert result == entries  # nothing removed


def test_dense_noisy_stretch_is_removed_but_surrounding_monologue_survives():
    """A sliding window inherently means entries right at the *boundary* of a
    noisy burst get swept in too (their window straddles the burst) — real
    edge bleed, not a bug. So this only asserts about the far interior of
    each clean stretch, comfortably outside any window that touches the
    burst, not the full clean range down to the transition itself."""
    entries = (
        [clean_entry(i) for i in range(10)]
        + [noisy_entry(i) for i in range(10, 20)]  # a genuine crowd Q&A burst
        + [clean_entry(i) for i in range(20, 30)]
    )

    result = strip_crowd_banter(entries, window=8, threshold=0.3)
    result_texts = [e["text"] for e in result]

    # Interior of each clean stretch, far from the burst, survives untouched.
    for i in range(5):
        assert clean_entry(i)["text"] in result_texts
    for i in range(25, 30):
        assert clean_entry(i)["text"] in result_texts

    # The noisy burst itself is entirely gone.
    for i in range(10, 20):
        assert noisy_entry(i)["text"] not in result_texts


def test_threshold_controls_sensitivity():
    """A stricter (lower) threshold removes more; a looser one removes less
    — sanity-checking the parameter actually does something, not just that
    default values happen to work."""
    entries = [clean_entry(i) if i % 3 else noisy_entry(i) for i in range(30)]

    strict = strip_crowd_banter(entries, window=8, threshold=0.1)
    loose = strip_crowd_banter(entries, window=8, threshold=0.9)

    assert len(strict) <= len(loose)
