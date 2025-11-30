import os
import requests
from transformers import pipeline

# ------------------------
# Local pipeline backend
# ------------------------
_pipe = None

def load_pipeline():
    """Lazy-load the local Hugging Face sentiment-analysis pipeline."""
    global _pipe
    if _pipe is None:
        _pipe = pipeline("sentiment-analysis")
    return _pipe


def analyze_batch_local(texts, margin=0.15):
    """
    Run local pipeline on a list of texts.
    Apply neutral margin: scores below margin -> NEUTRAL.
    """
    pipe = load_pipeline()
    raw = pipe(texts)

    processed = []
    for r in raw:
        label = r["label"].upper()
        score = float(r["score"])

        if score < margin:
            processed.append({"label": "NEUTRAL", "score": score})
        else:
            processed.append({"label": label, "score": score})

    return processed


# ------------------------
# API backend
# ------------------------
def analyze_batch_api(texts, margin=0.15):
    """
    Calls Hugging Face Inference API for sentiment analysis.
    Needs:
      - HF_API_TOKEN env var
      - optional HF_API_URL (else uses DistilBERT SST-2)
    """
    token = os.getenv("HF_API_TOKEN")
    if not token:
        return [{"label": "NEUTRAL", "score": 1.0, "error": "HF_API_TOKEN missing"}]

    url = os.getenv(
        "HF_API_URL",
        "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
    )

    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()  # expected: list[list[{label, score}]]

    processed = []
    for item in data:
        if not isinstance(item, list) or not item:
            processed.append({"label": "NEUTRAL", "score": 0.0})
            continue

        best = max(item, key=lambda x: x["score"])
        label = best["label"].upper()
        score = best["score"]

        if score < margin:
            processed.append({"label": "NEUTRAL", "score": score})
        else:
            processed.append({"label": label, "score": score})
    return processed
