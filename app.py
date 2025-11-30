import os
import time
import psutil
import gradio as gr
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from sentiment import analyze_batch_local, analyze_batch_api

# ------------------------
# Prometheus metrics
# ------------------------
SENT_REQ = Counter("sentiment_requests_total", "Total sentiment requests")
SENT_LAT = Histogram("sentiment_latency_seconds", "Sentiment request latency (s)")
SENT_CPU = Gauge("sentiment_cpu_percent", "CPU usage percent")
SENT_MEM = Gauge("sentiment_memory_mb", "Memory usage in MB")


@SENT_LAT.time()
def predict(text, neutral_margin, backend):
    """Main prediction function."""
    if not text.strip():
        return [{"label": "NEUTRAL", "score": 1.0}], "Please enter text."

    SENT_REQ.inc()
    start = time.perf_counter()

    if backend == "Local":
        results = analyze_batch_local([text], neutral_margin)
    else:
        results = analyze_batch_api([text], neutral_margin)

    latency_ms = (time.perf_counter() - start) * 1000

    proc = psutil.Process(os.getpid())
    SENT_CPU.set(proc.cpu_percent())
    SENT_MEM.set(proc.memory_info().rss / 1e6)

    label = results[0]["label"]
    score = results[0]["score"]

    md = f"""
### Result: **{label}**
Confidence: **{score:.3f}**

Latency: `{latency_ms:.1f} ms`  
CPU: `{proc.cpu_percent():.1f}%`  
Memory: `{SENT_MEM._value.get():.1f} MB`
"""
    return results, md


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Combined Sentiment Analysis (Local + API)")

        text = gr.Textbox(lines=5, label="Enter text")
        neutral_margin = gr.Slider(0.0, 0.5, value=0.15, label="Neutral margin")
        backend = gr.Radio(["Local", "API"], value="Local", label="Backend")
        output_json = gr.JSON()
        output_md = gr.Markdown()
        btn = gr.Button("Analyze", variant="primary")

        btn.click(predict, [text, neutral_margin, backend], [output_json, output_md])

    return demo


if __name__ == "__main__":
    start_http_server(8000)
    port = int(os.getenv("PORT", 7860))
    build_app().launch(server_name="0.0.0.0", server_port=port)
