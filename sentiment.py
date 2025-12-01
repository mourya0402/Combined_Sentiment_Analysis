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
# API backend (with error handling)
# ------------------------
def analyze_batch_api(texts, margin=0.15):
    """
    Calls Hugging Face Inference API for sentiment analysis.
    Needs:
      - HF_API_TOKEN env var
      - optional HF_API_URL (else uses DistilBERT SST-2)
    Returns a list of dicts with keys: label, score, (optional) error.
    """
    if not texts:
        return []

    token = os.getenv("HF_API_TOKEN")
    if not token:
        # No token: don't crash, just return an informative result
        return [{
            "label": "NEUTRAL",
            "score": 0.0,
            "error": "HF_API_TOKEN missing in environment"
        }]

    url = os.getenv(
        "HF_API_URL",
        "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )

    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        # Don't blindly raise; if it's bad, we'll encode it in the result
        if resp.status_code != 200:
            return [{
                "label": "NEUTRAL",
                "score": 0.0,
                "error": f"HF API HTTP {resp.status_code}: {resp.text[:200]}"
            }]

        data = resp.json()  # expected: list[list[{label, score}]]

        processed = []
        for item in data:
            if not isinstance(item, list) or not item:
                processed.append({"label": "NEUTRAL", "score": 0.0})
                continue

            best = max(item, key=lambda x: x.get("score", 0.0))
            label = best.get("label", "").upper()
            score = float(best.get("score", 0.0))

            if score < margin:
                processed.append({"label": "NEUTRAL", "score": score})
            else:
                processed.append({"label": label, "score": score})

        return processed

    except Exception as e:
        # Catch network / JSON / other errors and surface them in the result
        return [{
            "label": "NEUTRAL",
            "score": 0.0,
            "error": f"Exception calling HF API: {e}"
        }]
