# Adobe Persona-Driven PDF Section Extractor

This Dockerized app extracts and summarizes the most relevant PDF sections for a target persona and job-to-be-done, matching Adobe's challenge output format.

## Usage

1. **Build the Docker image**

2. **Prepare input**
- Place `input.json` and all required PDFs in an `input/` folder.

3. **Run the container**

4. **Review `output/output.json`**
- Extracted sections and refined summaries will appear matching Adobe’s schema.

## Requirements

- No internet at runtime
- CPU only, AMD64
- Input at `/app/input`; output at `/app/output`

## Output Schema

- Single `output.json` with:
- `"metadata"` (inputs, persona, timestamp)
- `"extracted_sections"` (top-5, ranked)
- `"subsection_analysis"` (refined text per section)
