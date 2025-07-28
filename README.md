**Overview & Approach**
This solution extracts the top 5 most relevant sections from a collection of PDFs based on a specified persona and job-to-be-done. For each relevant section, the system produces a human-like summary (“refined_text”). All outputs follow the Adobe challenge schema.

**Approach**
Section Extraction:
Each PDF is parsed using PyMuPDF to extract large, coherent content blocks—typically full paragraphs or multi-paragraph sections.
Section titles are generated using heading heuristics or the first sentence.

**Semantic Ranking:**
Each section’s text is combined with its title and encoded using a local Sentence Transformers model (“all-MiniLM-L6-v2”).
Query embeddings are generated from the persona and job description. Sections are ranked by cosine similarity to the query embedding.

**Dynamic Relevance Boosting:**
To increase alignment with any persona/job, dynamic keywords are extracted from the query (persona + job), and similarity scores are boosted for sections containing these keywords.

**Summarization:**
For each selected section, a summary (“refined_text”) is produced.
If the section is long, it is summarized using t5-small (via HuggingFace Transformers); otherwise, the full cleaned text is used directly. Summarization is guided with a prompt tailored to the persona/job context.

**Output Formatting:**
Results are output as a single output.json, matching Adobe’s schema:
metadata
extracted_sections (with section info & ranking)
subsection_analysis (with “refined_text” summaries)
**Models and Libraries Used**
PyMuPDF (fitz): PDF reading and text extraction.
sentence-transformers (“all-MiniLM-L6-v2”): For semantic embeddings and relevance ranking.
HuggingFace Transformers (“t5-small” + “sentencepiece”): For abstractive summarization of extracted content.
scikit-learn: For cosine similarity computation.
numpy, re, string: Utilities for text processing and analysis.

All models are pre-fetched into /app/model during Docker build.
The pipeline does not require internet access at runtime and works on CPU.

**How to Build & Run**
**Build the Docker image:**
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
**Prepare Input:**
Place your input.json and all PDFs referenced within it into an input/ folder in your project directory.
**Run the Docker container:**
docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" --network none pdf-outline-extractor:latest
(If using PowerShell, ensure ${PWD} resolves to your working directory.)

**Review the output:**
The result will be written to output/output.json in the required Adobe schema.


**OUTPUT FOR COLLECTION1 IN 1B :**
{
  "metadata": {
    "input_documents": [
      "South of France - Cities.pdf",
      "South of France - Cuisine.pdf",
      "South of France - History.pdf",
      "South of France - Restaurants and Hotels.pdf",
      "South of France - Things to Do.pdf",
      "South of France - Tips and Tricks.pdf",
      "South of France - Traditions and Culture.pdf"
    ],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2025-07-28T13:27:50.998958Z"
  },
  "extracted_sections": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "section_title": "Conclusion",
      "importance_rank": 1,
      "page_number": 9
    },
    {
      "document": "South of France - Things to Do.pdf",
      "section_title": "Family-Friendly Activities",
      "importance_rank": 2,
      "page_number": 9
    },
    {
      "document": "South of France - Cities.pdf",
      "section_title": "Aix-En-Provence: A City Of Art And Culture",
      "importance_rank": 3,
      "page_number": 8
    },
    {
      "document": "South of France - Restaurants and Hotels.pdf",
      "section_title": "Hotels",
      "importance_rank": 4,
      "page_number": 8
    },
    {
      "document": "South of France - History.pdf",
      "section_title": "Conclusion",
      "importance_rank": 5,
      "page_number": 12
    }
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "refined_text": "You'll be well; prepared for a comfortable and enjoyable trip. consider the season, planned activities, and the needs of both adults and children.",
      "page_number": 9
    },
    {
      "document": "South of France - Things to Do.pdf",
      "refined_text": "The south of France offers a variety of activities that are perfect for families with children. Theme Parks and Attractions ; Antibes: Visit Marineland for marine shows and an aquarium. Monteux: Spend a day at Parc Spirou, based on the famous comic book character.",
      "page_number": 9
    },
    {
      "document": "South of France - Cities.pdf",
      "refined_text": "Aix; en; Provence: A City of Art and Culture History Aix; en; Provence is known for its elegant architecture, vibrant cultural scene, and association with the painter Paul Cézanne. the city has been a center of art and learning for centuries, attracting artists, writers, and scholars.",
      "page_number": 8
    },
    {
      "document": "South of France - Restaurants and Hotels.pdf",
      "refined_text": "Hotels budget; friendly hotels ; Ibis Budget Nice Californie Lenval (Nice): a charming hotel located in a historic building, offering affordable rates and beautiful views of the Mediterranean. the artistic decor and convenient location make it a great choice for budget; conscious travelers.",
      "page_number": 8
    },
    {
      "document": "South of France - History.pdf",
      "refined_text": "The south of France offers a rich tapestry of history, culture, and architecture that is sure to captivate any traveler. from the ancient Roman ruins of Nîmes and Arles to the medieval fortresses of Carcassonne and Avignon, each city and town has its own unique story to tell.",
      "page_number": 12
    }
  ]
}
**BRIEF EXPLANATION OF CODE**
This solution processes the given PDFs by first splitting them into meaningful sections using a combination of text layout heuristics to detect headings and associated content blocks. It then uses powerful semantic embeddings from the sentence-transformers model (all-MiniLM-L6) to rank these sections by their relevance to the provided persona and job description.

To further align the ranking with the specific context of the task, the system dynamically extracts keywords from the persona and job text and slightly boosts the scores of sections containing these keywords.

For summarization, the approach leverages a pre-downloaded T5-small model, which generates natural, concise summaries of the relevant sections, guided explicitly by the persona and job context.

Finally, the code outputs the top five most relevant sections along with their human-readable summaries in the required JSON schema — all in an offline, distributed manner using Docker.
