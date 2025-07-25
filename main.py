import os
import json
import datetime
import re
from typing import List, Dict
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer


INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'
INPUT_JSON = os.path.join(INPUT_DIR, 'input.json')
MODEL_DIR = '/app/model'
ST_EMB_MODEL_NAME = 'all-MiniLM-L6-v2'
ST_EMB_PATH = os.path.join(MODEL_DIR, ST_EMB_MODEL_NAME)
T5_PATH = os.path.join(MODEL_DIR, 't5-small')

# Keywords to boost for persona/task relevance (group travel, activities, nightlife, packing, etc.)
KEYWORD_BOOSTS = [
    "group", "friends", "activities", "nightlife", "itinerary", "best places",
    "packing", "travel planner", "college", "trip", "family-friendly", "adventure",
    "restaurants", "hotels", "cuisine", "culture", "things to do", "tips"
]

def get_sentence_transformer():
    return SentenceTransformer(ST_EMB_PATH)

def get_t5_summarizer():
    tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
    model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
    return tokenizer, model

def cosine_sim(a, b):
    return float(cosine_similarity([a], [b])[0][0])

def embed(texts, model, batch_size=16):
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, convert_to_numpy=True, batch_size=batch_size)

def build_query_embedding(persona: Dict, job: Dict, model):
    combined_text = f"Persona: {persona.get('role', '')}. Task: {job.get('task', '')}"
    return embed([combined_text], model)[0]

def clean_title(title: str) -> str:
    if not title:
        return ''
    title = title.strip(":- \n\t")
    title = re.sub(r'\s+', ' ', title)
    return title.title()

def is_major_heading(line_text: str, line_spans: List[Dict]) -> bool:
    # Heuristic: bold + size >=12, OR all-caps short lines, OR title-case short line
    for span in line_spans:
        if (span.get('size', 0) >= 12) and ('bold' in span.get('font', '').lower()):
            return True
    if line_text.isupper() and 6 < len(line_text) < 80:
        return True
    if line_text.istitle() and len(line_text.split()) <= 9:
        return True
    return False

def extract_sections_from_pdf(filepath: str) -> List[Dict]:
    doc = fitz.open(filepath)
    sections = []
    last_heading_page = None
    last_section = None

    for page_no in range(doc.page_count):
        page = doc.load_page(page_no)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block['type'] != 0:
                continue
            for line in block["lines"]:
                line_text = " ".join(span["text"].strip() for span in line["spans"]).strip()
                if not line_text:
                    continue
                if is_major_heading(line_text, line["spans"]):
                    # New section start
                    # Append previous section only if it has text
                    if last_section and last_section["text"].strip():
                        sections.append(last_section)
                    last_section = {
                        "title": clean_title(line_text),
                        "text": "",
                        "page_number": page_no + 1
                    }
                    last_heading_page = page_no
                else:
                    # Append line text to current section if exists, else create default section
                    if last_section is None:
                        last_section = {
                            "title": f"Page {page_no+1} Untitled Section",
                            "text": "",
                            "page_number": page_no + 1,
                        }
                    last_section["text"] += line_text + " "
        # At page end: do nothing, wait for next heading or final flush
    # Append last section if exists
    if last_section and last_section["text"].strip():
        sections.append(last_section)

    # Fallback if no sections detected
    if not sections:
        full_text = "".join([page.get_text().strip() + "\n" for page in doc])
        sections.append({
            "title": "Document",
            "text": full_text.strip(),
            "page_number": 1
        })

    # Remove duplicates or empty sections
    seen_titles = set()
    cleaned_sections = []
    for sec in sections:
        t = sec["title"]
        if t.lower() in seen_titles or not sec["text"].strip():
            continue
        seen_titles.add(t.lower())
        cleaned_sections.append(sec)

    return cleaned_sections

def contains_keywords(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def boost_similarity_by_keywords(text: str, sim: float, keywords: List[str], boost_amount: float = 0.2) -> float:
    if contains_keywords(text, keywords):
        return sim + boost_amount
    return sim

def extract_refined_text_from_pdf_section(filename: str, page_number: int, heading: str, maxlen=1700) -> str:
    docfile = os.path.join(INPUT_DIR, filename)
    doc = fitz.open(docfile)
    page = doc.load_page(page_number - 1)
    text = page.get_text().replace('\r', '').replace('\u2022', '•').strip()
    # Clean copy for regex and match
    clean_text_for_search = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    clean_heading = re.sub(r'[^\w\s]', '', heading, flags=re.UNICODE)
    pattern = re.escape(clean_heading).replace(" ", r'\s*')
    heading_match = re.search(pattern, clean_text_for_search, re.IGNORECASE)

    content_start = 0
    if heading_match:
        # Find position in original text: approximate by char count
        idx_in_search = heading_match.end()
        # Map idx_in_search back to original text roughly by searching heading end substring
        pos = text.lower().find(heading.lower())
        content_start = pos + len(heading) if pos >= 0 else 0

    content = text[content_start:].strip()

    # Stop at next major heading (all-caps line) or double newline
    lines = content.split('\n')
    collected_lines = []
    cum_length = 0
    for line in lines:
        if line.strip() and line.strip().isupper() and len(line.strip()) > 6:
            break
        cum_length += len(line)
        if cum_length > maxlen:
            break
        collected_lines.append(line.strip())

    result = " ".join(collected_lines)
    # Clean lists/bullet points to semicolon separated
    result = re.sub(r'[\u2022\*\-]+', '; ', result)
    result = re.sub(r';{2,}', ';', result)
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()

    # Fallback to first few sentences if too short
    if len(result) < 70:
        sents = re.split(r'\. ', content)
        result = '. '.join(sents[:5]).strip()

    # Truncate to maxlen at sentence or semicolon boundary
    if len(result) > maxlen:
        idx = max(result.rfind('. ', 0, maxlen), result.rfind('; ', 0, maxlen))
        if idx > 0:
            result = result[:idx+1]
        else:
            result = result[:maxlen]

    return result[0].upper() + result[1:] if result else result

def summarize_with_t5(text, tokenizer, model, max_input_len=768, max_output_len=240, persona: str = "", task: str = ""):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if len(cleaned_text) > 3500:
        cleaned_text = cleaned_text[:3500]

    # Persona and task-aware prompt prefix
    prompt_prefix = f"For a {persona} tasked to {task}, summarize: "
    prompt = prompt_prefix + cleaned_text

    input_ids = tokenizer.encode(prompt,
                                 truncation=True,
                                 max_length=max_input_len,
                                 return_tensors="pt")
    summary_ids = model.generate(input_ids,
                                 max_length=max_output_len,
                                 num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    summary = summary[:1].upper() + summary[1:] if summary else ""
    summary = re.sub(r'\s*;\s*', '; ', summary)
    summary = re.sub(r';{2,}', ';', summary)
    summary = re.sub(r'\s+', ' ', summary)

    # Safety fallback for very short summary
    if len(summary) < 60:
        summary = cleaned_text[:max_output_len]
    return summary.strip()

def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        inp = json.load(f)
    persona = inp['persona']
    job = inp['job_to_be_done']
    docs = inp['documents']

    st_model = get_sentence_transformer()
    t5_tokenizer, t5_model = get_t5_summarizer()
    query_emb = build_query_embedding(persona, job, st_model)

    all_sections = []
    for doc in docs:
        filename = doc['filename']
        file_path = os.path.join(INPUT_DIR, filename)
        doc_sections = extract_sections_from_pdf(file_path)
        for sec in doc_sections:
            all_sections.append({
                'document': filename,
                'section_title': sec['title'],
                'section_text': sec['text'].strip(),
                'page_number': sec['page_number'],
            })

    # Calculate embeddings and similarity scores
    embedding_texts = [f"{s['section_title']}\n{s['section_text']}" for s in all_sections]
    section_embeds = embed(embedding_texts, st_model)
    for sec, emb in zip(all_sections, section_embeds):
        base_sim = cosine_sim(query_emb, emb)
        # Boost similarity if keywords found in title or text
        boosted_sim = boost_similarity_by_keywords(
            sec['section_title'] + " " + sec['section_text'], base_sim, KEYWORD_BOOSTS, boost_amount=0.25)
        sec['similarity'] = boosted_sim

    sorted_sections = sorted(all_sections, key=lambda s: s['similarity'], reverse=True)

    selected_sections = []
    docs_selected = set()
    for sec in sorted_sections:
        # Ensure diverse selection across documents
        if len(docs_selected) < len(docs):
            if sec['document'] in docs_selected:
                continue
            docs_selected.add(sec['document'])
        selected_sections.append(sec)
        if len(selected_sections) == 5:
            break

    metadata = {
        'input_documents': [doc['filename'] for doc in docs],
        'persona': persona.get('role', ''),
        'job_to_be_done': job.get('task', ''),
        'processing_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    }

    extracted_sections = []
    subsection_analysis = []

    for i, sec in enumerate(selected_sections, 1):
        extracted_sections.append({
            'document': sec['document'],
            'section_title': sec['section_title'],
            'importance_rank': i,
            'page_number': sec['page_number'],
        })
        # Extract full refined text from PDF page based on heading/title matching
        refined_text = extract_refined_text_from_pdf_section(sec['document'], sec['page_number'], sec['section_title'], maxlen=1700)

        # If text is long enough, and less than a threshold, keep as is; if longer, summarize to concise refined_text
        if 250 < len(refined_text) < 1200:
            summary = refined_text
        elif len(refined_text) >= 1200:
            summary = summarize_with_t5(refined_text, t5_tokenizer, t5_model, max_input_len=768, max_output_len=240, persona=metadata['persona'], task=metadata['job_to_be_done'])
        else:  # too short, fallback
            summary = summarize_with_t5(sec['section_text'], t5_tokenizer, t5_model, max_input_len=512, max_output_len=160, persona=metadata['persona'], task=metadata['job_to_be_done'])
        # Clean final summary
        summary = summary[0].upper() + summary[1:] if summary else ""
        summary = re.sub(r'\s*;\s*', '; ', summary)
        summary = re.sub(r';{2,}', ';', summary)
        summary = re.sub(r'\s+', ' ', summary)
        subsection_analysis.append({
            'document': sec['document'],
            'refined_text': summary,
            'page_number': sec['page_number'],
        })

    output = {
        'metadata': metadata,
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'output.json'), 'w', encoding='utf-8') as outf:
        json.dump(output, outf, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
