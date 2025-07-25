# Adobe Document Intelligence Challenge – Persona-driven Offline PDF Section Extractor

## Overview

This tool extracts and prioritizes the most relevant sections from a collection of PDF documents based on a given persona and job-to-be-done, as per the Adobe Document Intelligence Challenge offline requirements.

## Build & Run (Docker)

1. **Build (AMD64):**

2. **Prepare Inputs:**
- Place `input.json` and all referenced PDFs inside an input directory, e.g. `./input/`

3. **Run:**
docker run --rm
-v $(pwd)/input:/app/input
-v $(pwd)/output:/app/output
--network none
adobe-docintel:local

4. **Review Outputs:**
- See extracted results in `./output/`

## Outputs

- `output/output.json`: Final summary with top-5 global sections, sub-sections, and metadata.
- `output/[filename].json`: Per-PDF relevant sections.

## Requirements

- No internet access at runtime. All models are pre-packaged in the Docker image.
- AMD64 CPU, Python 3.10+
