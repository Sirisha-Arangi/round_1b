## Approach Explanation

### Section Extraction

Sections are extracted using page-level and heading-based heuristics. Bold, large font, ALL CAPS or Title Case short lines are treated as headings. Between headings (or as fallback, per page), contiguous text blocks are taken to ensure the summarizer receives a full narrative context, similar to Adobe's own.

### Section Ranking

Each candidate section's composite embedding (title + text) is ranked using cosine similarity vs. an embedding of the persona/task. This ensures that only the most relevant topics (city guides, tips, experiences, etc.) rise to the top.

### Refined Text Extraction

Upon selection, the code locates the exact heading in the nominated PDF page and extracts all content below, up to the next heading or the end of page. Bullet points are converted to semicolon-separated clauses, and paragraphs are merged as needed. If the block is long, it's summarized using T5-small. Otherwise, it is kept as extractive, since Adobe’s samples favor extractive summaries by default.

### Output & Compliance

A single `output.json` is produced, fully matching Adobe's challenge schema with three keys: metadata, extracted_sections, and subsection_analysis. All processing is CPU-only and fully offline; all models are pre-cached during Docker build.

This workflow was carefully engineered through side-by-side analysis to directly match Adobe's sample logic and outputs as closely as possible for both section selection and summary content.
