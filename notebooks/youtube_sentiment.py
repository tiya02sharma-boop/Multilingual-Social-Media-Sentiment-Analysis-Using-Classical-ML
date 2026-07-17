"""
Fetch comments from a YouTube video and run them through the sentiment
analysis pipeline served at /predict/batch (from app.py / inference.py).

Usage:
    python youtube_sentiment.py <video_id_or_url> [--max-comments 200] [--out results.csv]

Example:
    python youtube_sentiment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python youtube_sentiment.py dQw4w9WgXcQ --max-comments 500
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "notebooks" else SCRIPT_DIR

# Same "check both locations" pattern we used for the other API keys
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=False)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

# Where your FastAPI server is running (from app.py)
PREDICT_BATCH_URL = "http://127.0.0.1:8000/predict/batch"


# ---------------------------------------------------------------------------
# Step 1: Extract a plain video ID from either a raw ID or a full URL
# ---------------------------------------------------------------------------

def extract_video_id(video_input: str) -> str:
    video_input = video_input.strip()

    if re.fullmatch(r"[\w-]{11}", video_input):
        return video_input  # already looks like a bare video ID

    parsed = urlparse(video_input)

    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("/")[0]

    if "youtube.com" in parsed.netloc:
        # Shorts: youtube.com/shorts/VIDEO_ID
        if "/shorts/" in parsed.path:
            return parsed.path.split("/shorts/")[1].split("/")[0]

        # Regular video: youtube.com/watch?v=VIDEO_ID
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

    raise ValueError(
        f"Could not extract a video ID from: {video_input}\n"
        f"Paste the link exactly as it appears in your browser's address bar "
        f"(don't nest one URL inside another)."
    )


def _validate_video_id(video_id: str) -> str:
    if not re.fullmatch(r"[\w-]{11}", video_id):
        raise ValueError(
            f"Extracted value '{video_id}' doesn't look like a valid YouTube "
            f"video ID (should be exactly 11 characters). Check the URL you passed in."
        )
    return video_id


# ---------------------------------------------------------------------------
# Step 2: Fetch top-level comments, paginating until max_comments is hit
# ---------------------------------------------------------------------------

def fetch_youtube_comments(video_id: str, max_comments: int = 200) -> list[str]:
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            f"YOUTUBE_API_KEY not set (looked in {SCRIPT_DIR / '.env'} "
            f"and {PROJECT_ROOT / '.env'})"
        )

    comments = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "order": "relevance",
            "key": YOUTUBE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=15)

        if response.status_code == 403:
            # Common cause: comments disabled on this video, or quota exceeded
            print(f"[YouTube API error] 403: {response.json().get('error', {}).get('message')}")
            break

        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(top_comment)

        page_token = data.get("nextPageToken")
        if not page_token:
            break  # no more pages

    return comments[:max_comments]


# ---------------------------------------------------------------------------
# Step 3: Send comments to the sentiment pipeline in batches
# ---------------------------------------------------------------------------

def analyze_comments(comments: list[str], batch_size: int = 20) -> list[dict]:
    results = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i : i + batch_size]
        response = requests.post(
            PREDICT_BATCH_URL,
            json={"texts": batch},
            timeout=180,  # translation is now throttled, so batches take longer
        )
        response.raise_for_status()
        results.extend(response.json())
        print(f"  Analyzed {min(i + batch_size, len(comments))}/{len(comments)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YouTube comment sentiment analysis")
    parser.add_argument("video", help="YouTube video ID or full URL")
    parser.add_argument("--max-comments", type=int, default=200)
    parser.add_argument("--out", default="youtube_sentiment_results.csv")
    args = parser.parse_args()

    video_id = _validate_video_id(extract_video_id(args.video))
    print(f"Video ID: {video_id}")

    print("Fetching comments...")
    comments = fetch_youtube_comments(video_id, max_comments=args.max_comments)
    print(f"Fetched {len(comments)} comments")

    if not comments:
        print("No comments found — nothing to analyze.")
        sys.exit(0)

    print("Running sentiment analysis (make sure `uvicorn app:app --reload` is running)...")
    results = analyze_comments(comments)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} results to {args.out}")

    if "sentiment" in df.columns:
        print("\nSentiment breakdown:")
        print(df["sentiment"].value_counts(dropna=False))


if __name__ == "__main__":
    main()