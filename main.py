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

# Util to split text into large (paragraph-like) blocks
def split_paragraphs(text):
    # Split by two or more newlines, or large blocks
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 0]
    return paras

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
    return title.strip().title()

# Critical: Extract large content blocks!
def extract_sections_from_pdf(filepath: str) -> List[Dict]:
    doc = fitz.open(filepath)
    sections = []
    for page_no in range(doc.page_count):
        page = doc.load_page(page_no)
        text = page.get_text().strip()
        if not text or len(text.split()) < 20:
            continue
        # Try to split by "big" headings, else treat whole page as one block
        paras = split_paragraphs(text)
        # Try to merge short ones with long neighbor for context
        merged = []
        current_para = ""
        for p in paras:
            if len(current_para.split()) + len(p.split()) < 150:
                current_para += ' ' + p
            else:
                if current_para.strip():
                    merged.append(current_para.strip())
                current_para = p
        if current_para.strip():
            merged.append(current_para.strip())
        # Only use long paragraphs/blocks
        for mp in merged:
            if len(mp.split()) < 50:
                continue
            title = mp.split('\n')[0]
            if len(title) > 120 or not re.match(r'^[\w\s,.:;-]+$', title):
                title = f"Page {page_no+1} Block"
            sections.append({
                "title": clean_title(title)[:128],
                "text": mp,
                "page_number": page_no + 1
            })
    # fallback, if nothing: just one big doc
    if not sections:
        doctext = " ".join([page.get_text().strip() for page in doc])
        sections.append({
            "title": "Document",
            "text": doctext.strip(),
            "page_number": 1
        })
    return sections

def summarize_with_t5(text, tokenizer, model, max_input_len=768, max_output_len=240):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if len(cleaned_text) > 3500:
        cleaned_text = cleaned_text[:3500]
    prompt = "summarize: " + cleaned_text
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_len, return_tensors="pt")
    summary_ids = model.generate(input_ids, max_length=max_output_len, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    # Post-processing for readability
    summary = summary[:1].capitalize() + summary[1:] if summary else ""
    summary = re.sub(r'[\u2022\*\-]+', '; ', summary)
    summary = re.sub(r';{2,}', ';', summary)
    summary = re.sub(r'\s+', ' ', summary)
    # If summary is still too short or empty, fallback to cleaned block text
    if not summary or len(summary) < 60:
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
                'section_text': sec['text'],
                'page_number': sec['page_number']  # 1-based
            })

    # Similarity scoring (use combined title+content)
    embedding_texts = [f"{s['section_title']}\n{s['section_text']}" for s in all_sections]
    section_embeds = embed(embedding_texts, st_model)
    for sec, emb in zip(all_sections, section_embeds):
        sec['similarity'] = cosine_sim(query_emb, emb)
    all_sections_sorted = sorted(all_sections, key=lambda s: s['similarity'], reverse=True)

    # Section selection: keep from diverse docs if possible
    selected_sections = []
    docs_selected = set()
    for sec in all_sections_sorted:
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

    for idx, sec in enumerate(selected_sections, 1):
        extracted_sections.append({
            'document': sec['document'],
            'section_title': sec['section_title'],
            'importance_rank': idx,
            'page_number': sec['page_number'],
        })
        summary = summarize_with_t5(sec['section_text'], t5_tokenizer, t5_model)
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
    with open(os.path.join(OUTPUT_DIR, 'output.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
