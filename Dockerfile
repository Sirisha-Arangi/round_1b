# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF/pdfplumber and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpoppler-cpp-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/model

# Pre-download sentence-transformers model (all-MiniLM-L6-v2)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/app/model/all-MiniLM-L6-v2')"

# Pre-download T5-small model and tokenizer for offline use
RUN python3 -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
                 T5Tokenizer.from_pretrained('t5-small').save_pretrained('/app/model/t5-small'); \
                 T5ForConditionalGeneration.from_pretrained('t5-small').save_pretrained('/app/model/t5-small')"

# Copy your application code and docs
COPY main.py .
COPY README.md .
COPY approach_explanation.md .

# Command to run your application
CMD ["python", "main.py"]
