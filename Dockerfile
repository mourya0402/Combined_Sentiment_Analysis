FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tini \
        prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860 8000 9100

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD prometheus-node-exporter --web.listen-address=":9100" & python app.py
