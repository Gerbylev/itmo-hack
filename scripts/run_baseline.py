#!/usr/bin/env python3
"""Baseline for Multi-Lingual Video Fragment Retrieval Challenge.
Reproduces the notebook logic: chunk transcripts -> embed with multilingual-e5-large -> FAISS search.
"""

import pickle
import re
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────
BASE = '/root/data/video-rag'
WORK = '/root/output'

WINDOW      = 60.0
STEP        = 30.0
SKIP_HASHES = {'7d49c038'}
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K_RETRIEVE = 50
TOP_K_RESULT   = 5
MERGE_GAP      = 15.0

import os
os.makedirs(WORK, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────
def extract_hash(path: str):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def clean_text(text: str) -> str:
    return text.lower().strip()

def make_chunks(segments: list, window: float, step: float) -> list:
    if not segments:
        return []
    chunks = []
    total_end = segments[-1]['end']
    t_start   = segments[0]['start']
    while t_start < total_end:
        t_end       = t_start + window
        window_segs = [s for s in segments if s['end'] > t_start and s['start'] < t_end]
        if window_segs:
            text = ' '.join(clean_text(s['text']) for s in window_segs)
            chunks.append({
                'start': window_segs[0]['start'],
                'end':   window_segs[-1]['end'],
                'text':  text,
            })
        t_start += step
    return chunks

def merge_chunks(candidates, gap):
    by_video = {}
    for c in candidates:
        by_video.setdefault(c['video_hash'], []).append(c)
    merged = []
    for vh, chks in by_video.items():
        chks = sorted(chks, key=lambda x: x['start'])
        current = chks[0].copy()
        for nxt in chks[1:]:
            if nxt['start'] <= current['end'] + gap:
                current['end']   = max(current['end'], nxt['end'])
                current['score'] = max(current['score'], nxt['score'])
            else:
                merged.append(current)
                current = nxt.copy()
        merged.append(current)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

def retrieve(query, model, index, all_chunks, top_k_retrieve=50, top_k_result=5, merge_gap=15.0):
    query_vec = model.encode(
        [f"query: {query.lower().strip()}"],
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype('float32')
    scores, indices = index.search(query_vec, top_k_retrieve)
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        ch = all_chunks[idx]
        candidates.append({
            'video_hash': ch['video_hash'],
            'start': ch['start'], 'end': ch['end'],
            'score': float(score),
        })
    merged = merge_chunks(candidates, gap=merge_gap)
    return merged[:top_k_result]

def iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0

def evaluate(gt, results, ks=(1, 3, 5), iou_thresh=0.5):
    train_qids = set(gt.keys())
    sr = {k: [] for k in ks}
    vr = {k: [] for k in ks}
    for r in results:
        qid = r['query_id']
        if qid not in train_qids:
            continue
        gt_items = gt[qid]
        hits = r['hits']
        for k in ks:
            top_hits = hits[:k]
            pred_hashes = {h['video_hash'] for h in top_hits}
            gt_hashes   = {g['video_hash'] for g in gt_items}
            vr[k].append(int(bool(pred_hashes & gt_hashes)))
            sr_hit = 0
            for h in top_hits:
                for g in gt_items:
                    if h['video_hash'] == g['video_hash'] and iou(h['start'], h['end'], g['start'], g['end']) >= iou_thresh:
                        sr_hit = 1
                        break
                if sr_hit:
                    break
            sr[k].append(sr_hit)
    metrics = {}
    for k in ks:
        metrics[f'SR@{k}'] = np.mean(sr[k])
        metrics[f'VR@{k}'] = np.mean(vr[k])
    avg_sr = np.mean([metrics[f'SR@{k}'] for k in ks])
    avg_vr = np.mean([metrics[f'VR@{k}'] for k in ks])
    metrics['AvgSR'] = avg_sr
    metrics['AvgVR'] = avg_vr
    metrics['FinalScore'] = (avg_sr + avg_vr) / 2
    return metrics

# ── STEP 1: Build chunks ───────────────────────────────────────
print(f"Device: {DEVICE}")
print("Loading transcripts...")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)
print(f"Videos in pkl: {len(transcripts)}")

print("Building chunks...")
all_chunks = []
for key, segments in tqdm(transcripts.items()):
    video_hash = extract_hash(key)
    if video_hash is None or video_hash in SKIP_HASHES:
        continue
    for ch in make_chunks(segments, WINDOW, STEP):
        ch['video_hash'] = video_hash
        all_chunks.append(ch)
print(f"Total chunks: {len(all_chunks)}")

# ── STEP 2: Embed & index ──────────────────────────────────────
print("Loading model...")
model = SentenceTransformer('intfloat/multilingual-e5-large', device=DEVICE)

texts = [f"passage: {ch['text']}" for ch in all_chunks]
print(f"Encoding {len(texts)} chunks on {DEVICE}...")
embeddings = model.encode(texts, batch_size=128, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype('float32')
print(f"Embeddings shape: {embeddings.shape}")

print("Building FAISS index...")
dim = embeddings.shape[1]
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(embeddings)
print(f"Index built. Vectors: {index_flat.ntotal}")

faiss.write_index(index_flat, f'{WORK}/index.faiss')
with open(f'{WORK}/chunks.pkl', 'wb') as f:
    pickle.dump(all_chunks, f)

# ── STEP 3: Evaluate on train ──────────────────────────────────
train = pd.read_csv(f'{BASE}/train/train_qa.csv')
gt = {}
for _, row in train.iterrows():
    qid = row['question_id']
    vh = extract_hash(row['video_file'])
    gt.setdefault(qid, []).append({'video_hash': vh, 'start': row['start'], 'end': row['end']})

train_questions = train[['question_id', 'question_en']].drop_duplicates('question_id')

print(f"\nEvaluating on {len(train_questions)} train questions...")
train_results = []
for _, row in tqdm(train_questions.iterrows(), total=len(train_questions)):
    hits = retrieve(row['question_en'], model, index_flat, all_chunks,
                    TOP_K_RETRIEVE, TOP_K_RESULT, MERGE_GAP)
    train_results.append({'query_id': row['question_id'], 'hits': hits})

metrics = evaluate(gt, train_results)
print("\n=== Train metrics ===")
for k in (1, 3, 5):
    print(f"  SR@{k} = {metrics[f'SR@{k}']:.4f}    VR@{k} = {metrics[f'VR@{k}']:.4f}")
print(f"\n  AvgSR      = {metrics['AvgSR']:.4f}")
print(f"  AvgVR      = {metrics['AvgVR']:.4f}")
print(f"  FinalScore = {metrics['FinalScore']:.4f}")

# ── STEP 4: Generate submission ─────────────────────────────────
video_files = pd.read_csv(f'{BASE}/video_files.csv')
hash_to_filename = {}
for path in video_files['video_path']:
    h = extract_hash(path)
    filename = re.sub(r'\.\w+$', '', path.split('/')[-1])
    if h:
        hash_to_filename[h] = filename

test = pd.read_csv(f'{BASE}/test/test.csv')
print(f"\nRunning inference on {len(test)} test queries...")
results = []
for _, row in tqdm(test.iterrows(), total=len(test)):
    hits = retrieve(str(row['question']), model, index_flat, all_chunks,
                    TOP_K_RETRIEVE, TOP_K_RESULT, MERGE_GAP)
    results.append({'query_id': row['query_id'], 'hits': hits})

rows = []
for r in results:
    row = {'query_id': r['query_id']}
    for rank in range(1, 6):
        if rank <= len(r['hits']):
            h = r['hits'][rank - 1]
            fname = hash_to_filename.get(h['video_hash'], h['video_hash'])
            row[f'video_file_{rank}'] = fname
            row[f'start_{rank}'] = round(h['start'], 1)
            row[f'end_{rank}'] = round(h['end'], 1)
        else:
            row[f'video_file_{rank}'] = ''
            row[f'start_{rank}'] = 0.0
            row[f'end_{rank}'] = 0.0
    rows.append(row)

cols = ['query_id']
for rank in range(1, 6):
    cols += [f'video_file_{rank}', f'start_{rank}', f'end_{rank}']
submission = pd.DataFrame(rows, columns=cols)
submission.to_csv(f'{WORK}/submission.csv', index=False)
print(f"Submission saved: {WORK}/submission.csv  shape={submission.shape}")
print("Done!")
