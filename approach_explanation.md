## Approach Explanation

### Offline-First Document Intelligence

Our solution strictly follows the offline, CPU-only, persona-driven document analysis challenge.

**1. PDF Ingestion & Section Segmentation**

- **Parsing:** Uses `PyMuPDF` to process each PDF page as a dictionary of text blocks and spans.
- **Section Detection:** Heuristically segments documents into sections. Headings are identified by:
  - Bold or large font size spans,
  - Lines in ALL CAPS,
  - or lines suggestive of headers (short, ending with ":").
- Each segment includes the detected title, raw text, and the exact page number for traceability.

**2. Embedding Representation**

- Uses the compact `all-MiniLM-L6-v2` SentenceTransformer model (~82MB, pre-downloaded in Docker) to generate dense semantic embeddings.
- Both the composite "query" (persona + job) and each PDF section are embedded using this model.

**3. Similarity Scoring & Section Ranking**

- Computes cosine similarity between the query embedding and every section in every PDF.
- A global top-5 ranking is produced (across all PDFs) for the final summary.

**4. Result Enrichment**

- Each selected section is "refined" (truncated for length, avoiding mid-sentence cuts).
- Runs additional sub-section analysis, further splitting by blank lines or likely subheadings to provide more granular, actionable chunks.
- All output entries include provenance: file name, section title, page number, and similarity score.

**5. Offline and Reproducible**

- All model data is fetched at build time and baked into the Docker image for offline use.
- Only open-source CPU libraries are used.
- Handles 3–10 PDFs, arbitrary section granularity, and robustly falls back on documents where heading structure is unclear.

**6. Output**

- A global result file (`output.json`) and one per-PDF file, all schema-compliant.

This approach ensures transparency, auditability, and relevancy for downstream persona-driven automation or manual review.
