import json
import os
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# ---------------------------------------------------------------------------
# Setup — paths assume inference.py lives inside notebooks/, artifacts in ../data/
# ---------------------------------------------------------------------------

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# .env might sit next to inference.py (notebooks/.env) or at the project root
# (Comments Sentimental Analysis/.env) — check both so it works either way.
load_dotenv(NOTEBOOK_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=False)

MODEL_PATH = DATA_DIR / "best_sentiment_model.joblib"
VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = DATA_DIR / "label_encoder.joblib"

model = joblib.load(MODEL_PATH)
# This is a ColumnTransformer combining word-level + char-level TF-IDF,
# fit on a DataFrame with a "CommentText" column — NOT a plain TfidfVectorizer.
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.5-flash:generateContent"
)
_last_gemini_call = 0.0  # used to throttle requests, see translate_with_gemini

# ---------------------------------------------------------------------------
# Daily quota tracking — persisted to disk so the 20 requests/day limit is
# respected across multiple script runs and server restarts, not just within
# one process. Every attempt (including retries) counts, since RPD counts
# actual requests sent, not just successes.
# ---------------------------------------------------------------------------

GEMINI_DAILY_LIMIT = 20
QUOTA_FILE = NOTEBOOK_DIR / ".gemini_quota.json"


def _load_quota() -> dict:
    today = date.today().isoformat()
    if QUOTA_FILE.exists():
        try:
            data = json.loads(QUOTA_FILE.read_text())
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "count": 0}  # new day (or no file yet) — reset


def _save_quota(data: dict) -> None:
    try:
        QUOTA_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"[Quota tracker warning] could not save {QUOTA_FILE}: {e}")


def gemini_quota_remaining() -> int:
    return max(0, GEMINI_DAILY_LIMIT - _load_quota()["count"])


def _record_gemini_request() -> None:
    data = _load_quota()
    data["count"] += 1
    _save_quota(data)
    print(f"[Gemini quota] used {data['count']}/{GEMINI_DAILY_LIMIT} requests today")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# Translation only — Gemini API (Google AI Studio), free tier, no billing
# required. The prompt restricts it to translation, never classification.
# ---------------------------------------------------------------------------

def translate_with_gemini(text: str, max_retries: int = 2) -> Optional[str]:
    if not GEMINI_API_KEY:
        print(
            f"[Gemini translation error] GEMINI_API_KEY not set "
            f"(looked in {NOTEBOOK_DIR / '.env'} and {PROJECT_ROOT / '.env'})"
        )
        return None

    if gemini_quota_remaining() <= 0:
        print(f"[Gemini quota] daily limit of {GEMINI_DAILY_LIMIT} reached — skipping, using fallback")
        return None

    # Throttle to this project's actual observed limit: 5 RPM (confirmed via
    # AI Studio's rate limit dashboard — stricter than the general docs).
    global _last_gemini_call
    min_interval = 13.0  # 60s / 5 requests, with a small safety margin
    elapsed = time.monotonic() - _last_gemini_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_gemini_call = time.monotonic()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Translate the following text to English. "
                            "Return ONLY the translated text, no explanations, "
                            "no quotes, no additional commentary.\n\n"
                            f"Text: {text}"
                        )
                    }
                ],
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
            _record_gemini_request()  # every attempt counts against RPD, success or not

            # 503 (overloaded) and 429 (rate limited) are transient — worth retrying.
            # Anything else (401/403/400) won't fix itself, so fail immediately.
            if response.status_code in (503, 429) and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (2 ** attempt)
                print(
                    f"[Gemini translation warning] {response.status_code} on attempt "
                    f"{attempt}/{max_retries}, retrying in {wait}s"
                )
                time.sleep(wait)
                _last_gemini_call = time.monotonic()
                continue

            response.raise_for_status()
            data = response.json()
            translated = data["candidates"][0]["content"]["parts"][0]["text"]
            return translated.strip() if translated else None

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"[Gemini translation warning] attempt {attempt}/{max_retries} "
                    f"failed ({e}), retrying in {wait}s"
                )
                time.sleep(wait)
                _last_gemini_call = time.monotonic()

    print(f"[Gemini translation error] gave up after {max_retries} attempts: {last_error}")
    return None


def translate_batch_with_gemini(texts: List[str], max_retries: int = 2) -> Optional[List[str]]:
    """
    Translates many comments in a single Gemini call instead of one call per
    comment. This is the key speed fix: 20 comments = 1 request against the
    rate limit instead of 20, so far fewer 429s and far less total wait time.
    """
    if not GEMINI_API_KEY:
        print(
            f"[Gemini translation error] GEMINI_API_KEY not set "
            f"(looked in {NOTEBOOK_DIR / '.env'} and {PROJECT_ROOT / '.env'})"
        )
        return None
    if not texts:
        return []

    if gemini_quota_remaining() <= 0:
        print(f"[Gemini quota] daily limit of {GEMINI_DAILY_LIMIT} reached — skipping, using fallback")
        return None

    global _last_gemini_call
    min_interval = 13.0  # 60s / 5 requests, matches this project's actual RPM limit
    elapsed = time.monotonic() - _last_gemini_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_gemini_call = time.monotonic()

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = (
        "Translate each numbered comment below to English. "
        "Respond with ONLY a JSON array of strings, one per comment, in the "
        "same order, same length as the input. No explanations, no markdown "
        "code fences, no extra text — just the raw JSON array.\n\n"
        f"{numbered}"
    )

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=60)
            _record_gemini_request()  # every attempt counts against RPD, success or not

            if response.status_code in (503, 429) and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (2 ** attempt)
                print(
                    f"[Gemini batch translation warning] {response.status_code} on "
                    f"attempt {attempt}/{max_retries}, retrying in {wait}s"
                )
                time.sleep(wait)
                _last_gemini_call = time.monotonic()
                continue

            response.raise_for_status()
            data = response.json()
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            # Strip markdown code fences if the model added them anyway
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw[raw.find("["):]

            translations = json.loads(raw)

            if not isinstance(translations, list) or len(translations) != len(texts):
                raise ValueError(
                    f"expected {len(texts)} translations, got "
                    f"{len(translations) if isinstance(translations, list) else type(translations)}"
                )

            return [str(t).strip() for t in translations]

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"[Gemini batch translation warning] attempt {attempt}/{max_retries} "
                    f"failed ({e}), retrying in {wait}s"
                )
                time.sleep(wait)
                _last_gemini_call = time.monotonic()

    print(f"[Gemini batch translation error] gave up after {max_retries} attempts: {last_error}")
    return None


# ---------------------------------------------------------------------------
# Prediction — vectorizer needs a DataFrame with a "CommentText" column
# ---------------------------------------------------------------------------

def predict_sentiment(text: str) -> dict:
    df = pd.DataFrame({"CommentText": [text]})
    features = vectorizer.transform(df)

    pred_encoded = model.predict(features)[0]
    sentiment = label_encoder.inverse_transform([pred_encoded])[0]

    confidence = None
    confidence_type = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
        confidence_type = "predict_proba"
    elif hasattr(model, "decision_function"):
        margin = model.decision_function(features)[0]
        confidence = float(np.max(margin)) if hasattr(margin, "__len__") else float(margin)
        confidence_type = "decision_function_margin"  # raw SVM margin, not a true probability

    return {
        "sentiment": str(sentiment),
        "confidence": confidence,
        "confidence_type": confidence_type,
    }


# ---------------------------------------------------------------------------
# Full pipeline: detect language -> translate if needed -> predict
# ---------------------------------------------------------------------------

def run_pipeline(text: str) -> dict:
    # langdetect is informational only here — it reliably misreads romanized
    # Hinglish as English (or other Latin-script languages), since it was
    # trained on native-script corpora. So we don't let it gate translation.
    detected_lang = detect_language(text)

    translated_text = translate_with_gemini(text)
    method = "translated_via_gemini"

    if translated_text is None:
        # Gemini failed (likely quota) — fall back to the free unofficial
        # translator instead of giving up entirely.
        translated_text = translate_with_fallback(text)
        method = "translated_via_fallback"

    if translated_text is None:
        return {
            "text": text,
            "detected_language": detected_lang,
            "translated_text": None,
            "sentiment": None,
            "confidence": None,
            "confidence_type": None,
            "method": "translation_failed",
        }

    if translated_text.strip().lower() == text.strip().lower():
        method = "direct_english"

    result = predict_sentiment(translated_text)

    return {
        "text": text,
        "detected_language": detected_lang,
        "translated_text": translated_text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "confidence_type": result["confidence_type"],
        "method": method,
    }


# ---------------------------------------------------------------------------
# Fallback translation — free, unofficial Google Translate. Used only when
# Gemini fails an entire chunk (e.g. daily quota exhausted), so the pipeline
# keeps working instead of everything coming back translation_failed.
# ---------------------------------------------------------------------------

def translate_with_fallback(text: str) -> Optional[str]:
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated.strip() if translated else None
    except Exception as e:
        print(f"[Fallback translation error] {e}")
        return None


# ---------------------------------------------------------------------------
# Batch pipeline: translate many comments per Gemini call, then predict each
# ---------------------------------------------------------------------------

def run_pipeline_batch(texts: List[str], chunk_size: int = 20) -> List[dict]:
    """
    Translates in chunks of `chunk_size` per Gemini call. The persistent
    daily quota tracker (see gemini_quota_remaining) automatically stops
    attempting Gemini once 20 requests have been used today — across this
    run AND any earlier runs today — and every chunk after that goes
    straight to the free fallback translator instead.
    """
    results = [None] * len(texts)
    remaining_quota = gemini_quota_remaining()
    print(f"[Gemini quota] {remaining_quota}/{GEMINI_DAILY_LIMIT} requests remaining today")

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]

        translations = translate_batch_with_gemini(chunk)

        for offset, original_text in enumerate(chunk):
            idx = start + offset
            detected_lang = detect_language(original_text)

            if translations is None:
                # Either Gemini wasn't tried for this chunk, or it failed —
                # fall back to the free unofficial translator per comment.
                fallback_text = translate_with_fallback(original_text)
                if fallback_text is None:
                    results[idx] = {
                        "text": original_text,
                        "detected_language": detected_lang,
                        "translated_text": None,
                        "sentiment": None,
                        "confidence": None,
                        "confidence_type": None,
                        "method": "translation_failed",
                    }
                    continue

                pred = predict_sentiment(fallback_text)
                results[idx] = {
                    "text": original_text,
                    "detected_language": detected_lang,
                    "translated_text": fallback_text,
                    "sentiment": pred["sentiment"],
                    "confidence": pred["confidence"],
                    "confidence_type": pred["confidence_type"],
                    "method": "translated_via_fallback",
                }
                continue

            translated_text = translations[offset]
            method = (
                "translated_via_gemini"
                if translated_text.strip().lower() != original_text.strip().lower()
                else "direct_english"
            )
            pred = predict_sentiment(translated_text)
            results[idx] = {
                "text": original_text,
                "detected_language": detected_lang,
                "translated_text": translated_text,
                "sentiment": pred["sentiment"],
                "confidence": pred["confidence"],
                "confidence_type": pred["confidence_type"],
                "method": method,
            }

    return results


if __name__ == "__main__":
    samples = [
        "this video is amazing, loved it!",
        "yeh video bahut acha hai",
    ]
    for s in samples:
        print(run_pipeline(s))