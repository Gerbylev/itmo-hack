#!/usr/bin/env python3
"""Systematic experiments for Video Fragment Retrieval.
Tests different models, chunking params, and retrieval strategies.
Cleans up model cache between runs to save disk.
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

# ── Config ──────────────────────────────────────────────────────
BASE = '/root/data/video-rag'
WORK = '/root/output'
CACHE_DIR = '/root/.cache/huggingface'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}

os.makedirs(WORK, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────
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
    """Create chunks at multiple scales."""
    if not segments:
        return []
    all_chunks = []
    for window, step in windows_steps:
        chunks = make_chunks(segments, window, step)
        all_chunks.extend(chunks)
    return all_chunks

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
                current['end'] = max(current['end'], nxt['end'])
                current['score'] = max(current['score'], nxt['score'])
            else:
                merged.append(current)
                current = nxt.copy()
        merged.append(current)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

def retrieve(query_vec, index, all_chunks, top_k_retrieve=50, top_k_result=5, merge_gap=15.0):
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

def clear_cache():
    """Remove HF cache to free disk space."""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

# ── Load data once ──────────────────────────────────────────────
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
print(f"Train questions: {len(train_questions)}, Test queries: {len(test)}")

# ── Experiment definitions ──────────────────────────────────────
EXPERIMENTS = [
    # Experiment 1: Baseline reproduction
    {
        'name': 'baseline_e5large',
        'model': 'intfloat/multilingual-e5-large',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'chunking': [(60.0, 30.0)],
        'top_k_retrieve': 50,
        'merge_gap': 15.0,
        'batch_size': 128,
    },
    # Experiment 2: Smaller chunks for better SR
    {
        'name': 'e5large_small_chunks',
        'model': 'intfloat/multilingual-e5-large',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'chunking': [(30.0, 15.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'batch_size': 128,
    },
    # Experiment 3: Multi-scale chunking
    {
        'name': 'e5large_multiscale',
        'model': 'intfloat/multilingual-e5-large',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'chunking': [(30.0, 15.0), (60.0, 30.0), (120.0, 60.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'batch_size': 128,
    },
    # Experiment 4: BGE-M3 model
    {
        'name': 'bge_m3',
        'model': 'BAAI/bge-m3',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0), (60.0, 30.0), (120.0, 60.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'batch_size': 64,
    },
    # Experiment 5: E5-large-instruct
    {
        'name': 'e5large_instruct',
        'model': 'intfloat/multilingual-e5-large-instruct',
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'chunking': [(30.0, 15.0), (60.0, 30.0), (120.0, 60.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'batch_size': 128,
    },
    # Experiment 6: GTE-multilingual
    {
        'name': 'gte_multilingual',
        'model': 'Alibaba-NLP/gte-multilingual-base',
        'query_prefix': '',
        'passage_prefix': '',
        'chunking': [(30.0, 15.0), (60.0, 30.0), (120.0, 60.0)],
        'top_k_retrieve': 100,
        'merge_gap': 10.0,
        'batch_size': 128,
    },
]

# ── Run experiments ─────────────────────────────────────────────
results_log = []
prev_model_name = None

for exp in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp['name']}")
    print(f"{'='*60}")
    t0 = time.time()

    # Build chunks
    all_chunks = []
    for key, segments in transcripts.items():
        video_hash = extract_hash(key)
        if video_hash is None or video_hash in SKIP_HASHES:
            continue
        if len(exp['chunking']) == 1:
            w, s = exp['chunking'][0]
            chs = make_chunks(segments, w, s)
        else:
            chs = make_multiscale_chunks(segments, exp['chunking'])
        for ch in chs:
            ch['video_hash'] = video_hash
            all_chunks.append(ch)
    print(f"Chunks: {len(all_chunks)}")

    # Load model (reuse if same)
    if exp['model'] != prev_model_name:
        if prev_model_name is not None:
            del model
            clear_cache()
        print(f"Loading model: {exp['model']}...")
        try:
            model = SentenceTransformer(exp['model'], device=DEVICE, trust_remote_code=True)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            results_log.append({'name': exp['name'], 'error': str(e)})
            prev_model_name = None
            clear_cache()
            continue
        prev_model_name = exp['model']

    # Encode passages
    texts = [f"{exp['passage_prefix']}{ch['text']}" for ch in all_chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=exp['batch_size'], show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True).astype('float32')

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
        hits = retrieve(qvec, index, all_chunks, exp['top_k_retrieve'], 5, exp['merge_gap'])
        train_results.append({'query_id': row['question_id'], 'hits': hits})

    metrics = evaluate(gt, train_results)
    elapsed = time.time() - t0

    print(f"\n--- {exp['name']} ---")
    for k in (1, 3, 5):
        print(f"  SR@{k}={metrics[f'SR@{k}']:.4f}  VR@{k}={metrics[f'VR@{k}']:.4f}")
    print(f"  AvgSR={metrics['AvgSR']:.4f}  AvgVR={metrics['AvgVR']:.4f}  FinalScore={metrics['FinalScore']:.4f}")
    print(f"  Time: {elapsed:.0f}s  Chunks: {len(all_chunks)}")

    exp_result = {
        'name': exp['name'],
        'model': exp['model'],
        'chunking': str(exp['chunking']),
        'n_chunks': len(all_chunks),
        'time_s': round(elapsed, 1),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    results_log.append(exp_result)

    # Save best submission so far
    # Generate test submission for this experiment
    print("Generating test submission...")
    test_results = []
    for _, row in tqdm(test.iterrows(), total=len(test)):
        q = f"{exp['query_prefix']}{str(row['question']).lower().strip()}"
        qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qvec, index, all_chunks, exp['top_k_retrieve'], 5, exp['merge_gap'])
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

    # Cleanup embeddings
    del embeddings, index, all_chunks, texts
    torch.cuda.empty_cache()

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("EXPERIMENT SUMMARY")
print(f"{'='*60}")
summary = pd.DataFrame(results_log)
if 'FinalScore' in summary.columns:
    summary = summary.sort_values('FinalScore', ascending=False)
    print(summary[['name', 'SR@1', 'SR@5', 'VR@1', 'VR@5', 'AvgSR', 'AvgVR', 'FinalScore', 'n_chunks', 'time_s']].to_string(index=False))
    best = summary.iloc[0]
    print(f"\nBEST: {best['name']} with FinalScore={best['FinalScore']:.4f}")

# Save summary
summary.to_csv(f'{WORK}/experiment_summary.csv', index=False)
with open(f'{WORK}/experiment_summary.json', 'w') as f:
    json.dump(results_log, f, indent=2)
print(f"\nSummary saved to {WORK}/experiment_summary.csv")
print("All submissions saved.")
