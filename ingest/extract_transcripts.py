"""
Layer 1: Transcript Extraction

Pulls transcripts for the selected Sharan Hegde videos, chunks them into
400-word segments with 50-word overlap, tags each chunk with metadata
(video_id, title, start_time, source_url), and writes the result to
data/chunks.json for the embedding layer (Layer 2) to consume.
"""

import json
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "chunks.json"

VIDEOS = [
    {
        "video_id": "IgjhqPgwwGI",
        "title": "Everything You Need to Know About Money in Your 20s | 291-Minute Masterclass | Ft. Sharan Hegde",
        "source": "5hr_seminar",
    },
    {"video_id": "GjfjqfqDzCg", "title": "How To Invest For Early Retirement | How to RETIRE in YOUR 30s | Finance With Sharan", "source": "playlist"},
    {"video_id": "qsibNjO2Cvc", "title": "How to WIN Your DIVORCE Financially | Finance With Sharan", "source": "playlist"},
    {"video_id": "Iq_AftOxqMY", "title": "Reduce Your ELECTRICITY BILL by 90% | Finance With Sharan", "source": "playlist"},
    {"video_id": "DET3gmXW1uY", "title": "This HEALTH MISTAKE Will Cost You CRORES | Health Finance | Finance with Sharan", "source": "playlist"},
    {"video_id": "eNmWnAl0U80", "title": "SAVE 64% While SHOPPING Using These HACKS | Finance With Sharan", "source": "playlist"},
    {"video_id": "YiHop4ooqKQ", "title": "STOP FALLING For These FINANCE SCAMS- Part 1 | Finance With Sharan", "source": "playlist"},
    {"video_id": "RoiVyNe_qy8", "title": "STOP PAYING for TRAVELLING in India | Finance With Sharan", "source": "playlist"},
    {"video_id": "Ab4Kzz3kWUk", "title": "The ULTIMATE Guide to SAVING TAXES Through FAMILY | Finance With Sharan", "source": "playlist"},
    {"video_id": "uUdM-kiCOvA", "title": "Can You AFFORD to have a CHILD? | Finance with Sharan", "source": "playlist"},
    {"video_id": "CnM5E-frf8s", "title": "SAVE Yourself From These FINANCIAL SCAMS- Part 2 | Finance With Sharan", "source": "playlist"},
    {"video_id": "DcNToAAISc4", "title": "Why all Finfluencers won't get BANNED | SEBI New Rules", "source": "playlist"},
    {"video_id": "Y3cKly68pWI", "title": "Which Employee Spends The Most? | Assumptions vs Actual | Finance With Sharan", "source": "playlist"},
]

# Excluded per claude.md's "do NOT use" rule (guest interview / multi-speaker panel,
# not Sharan teaching solo):
#   ngY_m_9gwCk - "This is Why Men Dominate in Finance | ... | Monika Halan | Ep. 45"
#   EsMGwG_H2qA - "Budget 2025 Simplified: ... Experts Break It Down"


def fetch_transcript(video_id: str):
    fetched = YouTubeTranscriptApi().fetch(video_id)
    return [{"text": s.text, "start": s.start} for s in fetched]


def chunk_transcript(entries, video_id: str, title: str):
    """Group transcript entries into ~CHUNK_SIZE_WORDS-word chunks with overlap,
    tagging each chunk with the timestamp of its first word."""
    words = []  # (word, start_time) pairs, flattened across all transcript entries
    for entry in entries:
        for word in entry["text"].split():
            words.append((word, entry["start"]))

    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    chunks = []
    for i in range(0, len(words), step):
        window = words[i:i + CHUNK_SIZE_WORDS]
        if not window:
            continue
        start_time = int(window[0][1])
        chunks.append({
            "video_id": video_id,
            "title": title,
            "start_time": start_time,
            "source_url": f"https://youtube.com/watch?v={video_id}&t={start_time}",
            "text": " ".join(word for word, _ in window),
        })
        if i + CHUNK_SIZE_WORDS >= len(words):
            break
    return chunks


def main():
    DATA_DIR.mkdir(exist_ok=True)
    all_chunks = []

    for video in VIDEOS:
        video_id, title = video["video_id"], video["title"]
        print(f"Fetching transcript: {title} ({video_id})")
        try:
            entries = fetch_transcript(video_id)
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"  Skipped — {e}")
            continue

        chunks = chunk_transcript(entries, video_id, title)
        print(f"  {len(entries)} transcript segments -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(all_chunks)} chunks from {len(VIDEOS)} videos to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
