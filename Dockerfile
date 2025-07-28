# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpoppler-cpp-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model

RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/app/model/all-MiniLM-L6-v2')"
RUN python3 -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; T5Tokenizer.from_pretrained('t5-small').save_pretrained('/app/model/t5-small'); T5ForConditionalGeneration.from_pretrained('t5-small').save_pretrained('/app/model/t5-small')"

COPY main.py .
COPY README.md .


CMD ["python", "main.py"]
