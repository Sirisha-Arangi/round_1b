import os
import json
import datetime
import re
import string
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

def split_paragraphs(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 0]

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

def extract_keywords(text: str, top_n: int = 12):
    stopwords = set([
        'the', 'and', 'is', 'in', 'of', 'to', 'a', 'for', 'with', 'on', 'by', 'as', 'at', 'this',
        'that', 'from', 'or', 'an', 'be', 'are', 'it', 'your', 'use', 'can', 'will', 'you', 'but',
        'not', 'have', 'has', 'we', 'they', 'its', 'which', 'if', 'who', 'what', 'how'
    ])
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    freqs = {}
    for word in words:
        if word not in stopwords and len(word) > 2:
            freqs[word] = freqs.get(word, 0) + 1
    keywords = sorted(freqs, key=freqs.get, reverse=True)[:top_n]
    return set(keywords)

def boost_similarity_by_keywords(text: str, base_sim: float, keywords: set, boost_amount: float = 0.25):
    text_lower = text.lower()
    if any(kw in text_lower for kw in keywords):
        return base_sim + boost_amount
    return base_sim

def extract_sections_from_pdf(filepath: str) -> List[Dict]:
    doc = fitz.open(filepath)
    sections = []
    for page_no in range(doc.page_count):
        page = doc.load_page(page_no)
        text = page.get_text().strip()
        if not text or len(text.split()) < 20:
            continue
        paras = split_paragraphs(text)
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
    if not sections:
        doctext = " ".join([page.get_text().strip() for page in doc])
        sections.append({
            "title": "Document",
            "text": doctext.strip(),
            "page_number": 1
        })
    return sections

def summarize_with_t5(text, tokenizer, model, max_input_len=768, max_output_len=240, persona: str = "", task: str = ""):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if len(cleaned_text) > 3500:
        cleaned_text = cleaned_text[:3500]
    prompt = f"For a {persona} tasked to {task}, summarize: {cleaned_text}"
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_len, return_tensors="pt")
    summary_ids = model.generate(input_ids, max_length=max_output_len, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    summary = summary[:1].capitalize() + summary[1:] if summary else ""
    summary = re.sub(r'[\u2022\*\-]+', '; ', summary)
    summary = re.sub(r';{2,}', ';', summary)
    summary = re.sub(r'\s+', ' ', summary)
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

    # Extract dynamic keywords from persona and job to boost relevance
    dynamic_keywords = extract_keywords(f"{persona} {job}", top_n=15)

    all_sections = []
    for doc in docs:
        filename = doc['filename']
        doc_sections = extract_sections_from_pdf(os.path.join(INPUT_DIR, filename))
        for sec in doc_sections:
            all_sections.append({
                'document': filename,
                'section_title': sec['title'],
                'section_text': sec['text'],
                'page_number': sec['page_number']
            })

    embedding_texts = [f"{s['section_title']}\n{s['section_text']}" for s in all_sections]
    section_embeds = embed(embedding_texts, st_model)
    for sec, emb in zip(all_sections, section_embeds):
        base_sim = cosine_sim(query_emb, emb)
        combined_text = sec['section_title'] + " " + sec['section_text']
        boosted_sim = boost_similarity_by_keywords(combined_text, base_sim, dynamic_keywords, boost_amount=0.25)
        sec['similarity'] = boosted_sim

    all_sections_sorted = sorted(all_sections, key=lambda s: s['similarity'], reverse=True)

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
        'persona': persona if isinstance(persona, str) else persona.get('role', ''),
        'job_to_be_done': job if isinstance(job, str) else job.get('task', ''),
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
        summary = summarize_with_t5(
            sec['section_text'], t5_tokenizer, t5_model,
            persona=metadata['persona'], task=metadata['job_to_be_done']
        )
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
    output_path = os.path.join(OUTPUT_DIR, 'output.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
