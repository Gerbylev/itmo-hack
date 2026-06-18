#!/usr/bin/env python3
"""Experiments V2: Combine best findings.
- BGE-M3 had best VR (0.6367)
- Small chunks had best SR (0.1098)
- Try: BGE-M3 + small chunks, BGE-M3 + adaptive merge, two-stage retrieval
"""

import pickle
import re
import os
import shutil
import json
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

BASE = '/root/data/video-rag'
WORK = '/root/output'
CACHE_DIR = '/root/.cache/huggingface'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def clean_text(text):
    return text.lower().strip()

def make_chunks(segments, window, step):
    if not segments:
        return []
    chunks = []
    total_end = segments[-1]['end']
    t_start = segments[0]['start']
    while t_start < total_end:
        t_end = t_start + window
        window_segs = [s for s in segments if s['end'] > t_start and s['start'] < t_end]
        if window_segs:
            text = ' '.join(clean_text(s['text']) for s in window_segs)
            chunks.append({
                'start': window_segs[0]['start'],
                'end': window_segs[-1]['end'],
                'text': text,
            })
        t_start += step
    return chunks

def make_multiscale_chunks(segments, windows_steps):
    all_chunks = []
    for window, step in windows_steps:
        all_chunks.extend(make_chunks(segments, window, step))
    return all_chunks

def merge_chunks_v2(candidates, gap, shrink_factor=0.0):
    """
    Improved merging with optional shrink.
    shrink_factor: shrink merged segment boundaries towards highest-scoring chunk center.
    """
    by_video = {}
    for c in candidates:
        by_video.setdefault(c['video_hash'], []).append(c)
    merged = []
    for vh, chks in by_video.items():
        chks = sorted(chks, key=lambda x: x['start'])
        current = chks[0].copy()
        best_chunk = chks[0].copy()
        for nxt in chks[1:]:
            if nxt['start'] <= current['end'] + gap:
                current['end'] = max(current['end'], nxt['end'])
                if nxt['score'] > current['score']:
                    current['score'] = nxt['score']
                    best_chunk = nxt.copy()
            else:
                # Apply shrink
                if shrink_factor > 0:
                    bc_center = (best_chunk['start'] + best_chunk['end']) / 2
                    bc_half = (best_chunk['end'] - best_chunk['start']) / 2
                    cur_half = (current['end'] - current['start']) / 2
                    new_half = cur_half * (1 - shrink_factor) + bc_half * shrink_factor
                    current['start'] = max(current['start'], bc_center - new_half)
                    current['end'] = min(current['end'], bc_center + new_half)
                merged.append(current)
                current = nxt.copy()
                best_chunk = nxt.copy()
        if shrink_factor > 0:
            bc_center = (best_chunk['start'] + best_chunk['end']) / 2
            bc_half = (best_chunk['end'] - best_chunk['start']) / 2
            cur_half = (current['end'] - current['start']) / 2
            new_half = cur_half * (1 - shrink_factor) + bc_half * shrink_factor
            current['start'] = max(current['start'], bc_center - new_half)
            current['end'] = min(current['end'], bc_center + new_half)
        merged.append(current)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

def retrieve_v2(query_vec, index, all_chunks, top_k_retrieve=100, top_k_result=5,
                merge_gap=10.0, shrink_factor=0.0):
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
    merged = merge_chunks_v2(candidates, gap=merge_gap, shrink_factor=shrink_factor)
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
            gt_hashes = {g['video_hash'] for g in gt_items}
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

# ── Load data ───────────────────────────────────────────────────
print("Loading data...")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)

train = pd.read_csv(f'{BASE}/train/train_qa.csv')
gt = {}
for _, row in train.iterrows():
    qid = row['question_id']
    vh = extract_hash(row['video_file'])
    gt.setdefault(qid, []).append({'video_hash': vh, 'start': row['start'], 'end': row['end']})

train_questions = train[['question_id', 'question_en']].drop_duplicates('question_id')

video_files = pd.read_csv(f'{BASE}/video_files.csv')
hash_to_filename = {}
for path in video_files['video_path']:
    h = extract_hash(path)
    filename = re.sub(r'\.\w+$', '', path.split('/')[-1])
    if h:
        hash_to_filename[h] = filename

test = pd.read_csv(f'{BASE}/test/test.csv')

# ── Build chunk sets ────────────────────────────────────────────
def build_chunks(chunking_config):
    all_chunks = []
    for key, segments in transcripts.items():
        video_hash = extract_hash(key)
        if video_hash is None or video_hash in SKIP_HASHES:
            continue
        if len(chunking_config) == 1:
            w, s = chunking_config[0]
            chs = make_chunks(segments, w, s)
        else:
            chs = make_multiscale_chunks(segments, chunking_config)
        for ch in chs:
            ch['video_hash'] = video_hash
            all_chunks.append(ch)
    return all_chunks

# ── Experiments ─────────────────────────────────────────────────
EXPERIMENTS = [
    # BGE-M3 with small chunks
    {
        'name': 'bge_m3_small',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.0,
        'batch_size': 64,
    },
    # BGE-M3 small chunks + shrink
    {
        'name': 'bge_m3_small_shrink30',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.3,
        'batch_size': 64,
    },
    # BGE-M3 small+medium chunks
    {
        'name': 'bge_m3_small_medium',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0), (60.0, 30.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.0,
        'batch_size': 64,
    },
    # BGE-M3 small+medium + shrink
    {
        'name': 'bge_m3_sm_shrink30',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0), (60.0, 30.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.3,
        'batch_size': 64,
    },
    # BGE-M3 small chunks, tighter merge
    {
        'name': 'bge_m3_small_gap5',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0)],
        'top_k_retrieve': 100,
        'merge_gap': 5.0,
        'shrink': 0.0,
        'batch_size': 64,
    },
    # BGE-M3 small chunks, no merge
    {
        'name': 'bge_m3_small_gap0',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0)],
        'top_k_retrieve': 100,
        'merge_gap': 0.0,
        'shrink': 0.0,
        'batch_size': 64,
    },
    # BGE-M3 very small chunks (15s/7.5s)
    {
        'name': 'bge_m3_tiny',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(15.0, 7.5)],
        'top_k_retrieve': 150,
        'merge_gap': 5.0,
        'shrink': 0.0,
        'batch_size': 64,
    },
    # BGE-M3 medium (45s/20s) - balance SR and VR
    {
        'name': 'bge_m3_medium45',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(45.0, 20.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.0,
        'batch_size': 64,
    },
    # E5-large small chunks with shrink
    {
        'name': 'e5large_small_shrink30',
        'model': 'intfloat/multilingual-e5-large',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'chunking': [(30.0, 15.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.3,
        'batch_size': 128,
    },
    # E5-large with medium chunks
    {
        'name': 'e5large_medium45',
        'model': 'intfloat/multilingual-e5-large',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'chunking': [(45.0, 20.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'shrink': 0.0,
        'batch_size': 128,
    },
]

results_log = []
prev_model_name = None
prev_chunking = None
cached_embeddings = None
cached_chunks = None

for exp in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp['name']}")
    print(f"{'='*60}")
    t0 = time.time()

    # Build chunks (cache if same chunking)
    chunking_key = str(exp['chunking'])
    if chunking_key != prev_chunking:
        all_chunks = build_chunks(exp['chunking'])
        prev_chunking = chunking_key
        cached_chunks = all_chunks
        cached_embeddings = None  # Force re-encode
    else:
        all_chunks = cached_chunks
    print(f"Chunks: {len(all_chunks)}")

    # Load model
    if exp['model'] != prev_model_name:
        if prev_model_name is not None:
            del model
            torch.cuda.empty_cache()
            import gc; gc.collect()
        print(f"Loading model: {exp['model']}...")
        model = SentenceTransformer(exp['model'], device=DEVICE, trust_remote_code=True)
        prev_model_name = exp['model']
        cached_embeddings = None  # Force re-encode

    # Encode (cache if same model+chunking)
    if cached_embeddings is None:
        texts = [f"{exp['passage_prefix']}{ch['text']}" for ch in all_chunks]
        print(f"Encoding {len(texts)} chunks...")
        cached_embeddings = model.encode(texts, batch_size=exp['batch_size'], show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    embeddings = cached_embeddings

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Evaluate on train
    print("Evaluating on train...")
    train_results = []
    for _, row in tqdm(train_questions.iterrows(), total=len(train_questions)):
        q = f"{exp['query_prefix']}{row['question_en'].lower().strip()}"
        qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve_v2(qvec, index, all_chunks, exp['top_k_retrieve'], 5,
                          exp['merge_gap'], exp['shrink'])
        train_results.append({'query_id': row['question_id'], 'hits': hits})

    metrics = evaluate(gt, train_results)
    elapsed = time.time() - t0

    print(f"\n--- {exp['name']} ---")
    for k in (1, 3, 5):
        print(f"  SR@{k}={metrics[f'SR@{k}']:.4f}  VR@{k}={metrics[f'VR@{k}']:.4f}")
    print(f"  AvgSR={metrics['AvgSR']:.4f}  AvgVR={metrics['AvgVR']:.4f}  FinalScore={metrics['FinalScore']:.4f}")
    print(f"  Time: {elapsed:.0f}s")

    exp_result = {
        'name': exp['name'],
        'model': exp['model'],
        'chunking': str(exp['chunking']),
        'n_chunks': len(all_chunks),
        'merge_gap': exp['merge_gap'],
        'shrink': exp['shrink'],
        'time_s': round(elapsed, 1),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    results_log.append(exp_result)

    # Generate test submission
    print("Generating test submission...")
    test_results = []
    for _, row in tqdm(test.iterrows(), total=len(test)):
        q = f"{exp['query_prefix']}{str(row['question']).lower().strip()}"
        qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve_v2(qvec, index, all_chunks, exp['top_k_retrieve'], 5,
                          exp['merge_gap'], exp['shrink'])
        test_results.append({'query_id': row['query_id'], 'hits': hits})

    rows = []
    for r in test_results:
        row_d = {'query_id': r['query_id']}
        for rank in range(1, 6):
            if rank <= len(r['hits']):
                h = r['hits'][rank - 1]
                fname = hash_to_filename.get(h['video_hash'], h['video_hash'])
                row_d[f'video_file_{rank}'] = fname
                row_d[f'start_{rank}'] = round(h['start'], 1)
                row_d[f'end_{rank}'] = round(h['end'], 1)
            else:
                row_d[f'video_file_{rank}'] = ''
                row_d[f'start_{rank}'] = 0.0
                row_d[f'end_{rank}'] = 0.0
        rows.append(row_d)
    cols = ['query_id']
    for rank in range(1, 6):
        cols += [f'video_file_{rank}', f'start_{rank}', f'end_{rank}']
    submission = pd.DataFrame(rows, columns=cols)
    submission.to_csv(f'{WORK}/submission_{exp["name"]}.csv', index=False)

    del index
    torch.cuda.empty_cache()

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("V2 EXPERIMENT SUMMARY")
print(f"{'='*60}")
summary = pd.DataFrame(results_log)
summary = summary.sort_values('FinalScore', ascending=False)
print(summary[['name', 'SR@1', 'SR@5', 'VR@1', 'VR@5', 'AvgSR', 'AvgVR', 'FinalScore', 'n_chunks']].to_string(index=False))
best = summary.iloc[0]
print(f"\nBEST: {best['name']} with FinalScore={best['FinalScore']:.4f}")
summary.to_csv(f'{WORK}/experiment_v2_summary.csv', index=False)
with open(f'{WORK}/experiment_v2_summary.json', 'w') as f:
    json.dump(results_log, f, indent=2)
print("Done!")
