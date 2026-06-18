#!/usr/bin/env python3
"""Evaluate fine-tuned models with various retrieval configs and generate best submission."""

import pickle, re, os, numpy as np, pandas as pd, faiss, torch, time, gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def clean_text(text):
    return text.lower().strip()

def make_chunks(segments, window, step):
    if not segments: return []
    chunks, total_end, t_start = [], segments[-1]['end'], segments[0]['start']
    while t_start < total_end:
        t_end = t_start + window
        ws = [s for s in segments if s['end'] > t_start and s['start'] < t_end]
        if ws:
            chunks.append({'start': ws[0]['start'], 'end': ws[-1]['end'],
                          'text': ' '.join(clean_text(s['text']) for s in ws)})
        t_start += step
    return chunks

def build_chunks(chunking_config, transcripts):
    all_chunks = []
    for key, segments in transcripts.items():
        vh = extract_hash(key)
        if vh is None or vh in SKIP_HASHES: continue
        for w, s in chunking_config:
            for ch in make_chunks(segments, w, s):
                ch['video_hash'] = vh
                all_chunks.append(ch)
    return all_chunks

def merge_chunks_v2(candidates, gap, shrink_factor=0.0):
    by_video = {}
    for c in candidates:
        by_video.setdefault(c['video_hash'], []).append(c)
    merged = []
    for vh, chks in by_video.items():
        chks = sorted(chks, key=lambda x: x['start'])
        current, best = chks[0].copy(), chks[0].copy()
        for nxt in chks[1:]:
            if nxt['start'] <= current['end'] + gap:
                current['end'] = max(current['end'], nxt['end'])
                if nxt['score'] > current['score']:
                    current['score'] = nxt['score']
                    best = nxt.copy()
            else:
                if shrink_factor > 0:
                    bc = (best['start'] + best['end']) / 2
                    bh = (best['end'] - best['start']) / 2
                    ch = (current['end'] - current['start']) / 2
                    nh = ch * (1 - shrink_factor) + bh * shrink_factor
                    current['start'] = max(current['start'], bc - nh)
                    current['end'] = min(current['end'], bc + nh)
                merged.append(current)
                current, best = nxt.copy(), nxt.copy()
        if shrink_factor > 0:
            bc = (best['start'] + best['end']) / 2
            bh = (best['end'] - best['start']) / 2
            ch = (current['end'] - current['start']) / 2
            nh = ch * (1 - shrink_factor) + bh * shrink_factor
            current['start'] = max(current['start'], bc - nh)
            current['end'] = min(current['end'], bc + nh)
        merged.append(current)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

def retrieve(qvec, index, chunks, top_k=100, top_n=5, gap=10.0, shrink=0.0):
    scores, indices = index.search(qvec, top_k)
    cands = [{'video_hash': chunks[i]['video_hash'], 'start': chunks[i]['start'],
              'end': chunks[i]['end'], 'score': float(s)}
             for s, i in zip(scores[0], indices[0]) if i != -1]
    return merge_chunks_v2(cands, gap, shrink)[:top_n]

def iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0

def evaluate(gt, results, ks=(1, 3, 5)):
    sr, vr = {k: [] for k in ks}, {k: [] for k in ks}
    for r in results:
        if r['query_id'] not in gt: continue
        gi = gt[r['query_id']]
        for k in ks:
            th = r['hits'][:k]
            vr[k].append(int(bool({h['video_hash'] for h in th} & {g['video_hash'] for g in gi})))
            sh = 0
            for h in th:
                for g in gi:
                    if h['video_hash'] == g['video_hash'] and iou(h['start'], h['end'], g['start'], g['end']) >= 0.5:
                        sh = 1; break
                if sh: break
            sr[k].append(sh)
    m = {}
    for k in ks:
        m[f'SR@{k}'], m[f'VR@{k}'] = np.mean(sr[k]), np.mean(vr[k])
    m['AvgSR'] = np.mean([m[f'SR@{k}'] for k in ks])
    m['AvgVR'] = np.mean([m[f'VR@{k}'] for k in ks])
    m['FinalScore'] = (m['AvgSR'] + m['AvgVR']) / 2
    return m

# Load data
print("Loading data...")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)
train = pd.read_csv(f'{BASE}/train/train_qa.csv')
gt = {}
for _, row in train.iterrows():
    qid, vh = row['question_id'], extract_hash(row['video_file'])
    gt.setdefault(qid, []).append({'video_hash': vh, 'start': row['start'], 'end': row['end']})
tq = train[['question_id', 'question_en']].drop_duplicates('question_id')
vf = pd.read_csv(f'{BASE}/video_files.csv')
h2f = {}
for p in vf['video_path']:
    h = extract_hash(p); fn = re.sub(r'\.\w+$', '', p.split('/')[-1])
    if h: h2f[h] = fn
test = pd.read_csv(f'{BASE}/test/test.csv')

# Chunk configs to try
chunk_configs = {
    'sm': [(30.0, 15.0), (60.0, 30.0)],
    's': [(30.0, 15.0)],
    'sml': [(30.0, 15.0), (60.0, 30.0), (90.0, 45.0)],
    'sm_dense': [(30.0, 10.0), (60.0, 20.0)],
}

# Models to evaluate
MODELS = {
    'ft_e5_large': '/root/output/finetuned_e5_large',
    'ft_bge_m3': '/root/output/finetuned_bge_m3',
}

# Experiments
EXPERIMENTS = [
    # Best from baseline applied to finetuned
    {'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.7},
    {'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.5},
    {'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.3},
    {'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.8},
    {'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.9},
    # Different gaps
    {'chunks': 'sm', 'top_k': 100, 'gap': 5, 'shrink': 0.7},
    {'chunks': 'sm', 'top_k': 100, 'gap': 15, 'shrink': 0.7},
    {'chunks': 'sm', 'top_k': 100, 'gap': 20, 'shrink': 0.7},
    # Different top_k
    {'chunks': 'sm', 'top_k': 50, 'gap': 10, 'shrink': 0.7},
    {'chunks': 'sm', 'top_k': 200, 'gap': 10, 'shrink': 0.7},
    # Other chunk configs
    {'chunks': 's', 'top_k': 100, 'gap': 10, 'shrink': 0.7},
    {'chunks': 'sml', 'top_k': 100, 'gap': 10, 'shrink': 0.7},
    {'chunks': 'sm_dense', 'top_k': 100, 'gap': 10, 'shrink': 0.7},
]

results_log = []

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Loading {model_name} from {model_path}")
    print(f"{'='*60}")
    model = SentenceTransformer(model_path, device=DEVICE, trust_remote_code=True)

    # Pre-compute embeddings for each chunk config
    emb_cache = {}
    chunk_cache = {}
    for cname, cfg in chunk_configs.items():
        chunks = build_chunks(cfg, transcripts)
        chunk_cache[cname] = chunks
        texts = [ch['text'] for ch in chunks]
        print(f"  Encoding '{cname}' ({len(texts)} chunks)...")
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        emb_cache[cname] = emb

    # Pre-compute query embeddings
    print("  Encoding queries...")
    q_texts = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
    q_ids = [row['question_id'] for _, row in tq.iterrows()]
    q_embs = model.encode(q_texts, batch_size=64, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True).astype('float32')

    for exp in EXPERIMENTS:
        cname = exp['chunks']
        chunks = chunk_cache[cname]
        emb = emb_cache[cname]
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)

        tr = []
        for i, qid in enumerate(q_ids):
            qv = q_embs[i:i+1]
            hits = retrieve(qv, idx, chunks, exp['top_k'], 5, exp['gap'], exp['shrink'])
            tr.append({'query_id': qid, 'hits': hits})

        m = evaluate(gt, tr)
        name = f"{model_name}_{cname}_k{exp['top_k']}_g{exp['gap']}_s{int(exp['shrink']*10)}"
        print(f"  {name}: SR@1={m['SR@1']:.4f} VR@1={m['VR@1']:.4f} FS={m['FinalScore']:.4f}")
        results_log.append({'model': model_name, 'name': name, **exp,
                           **{k: round(v,4) for k,v in m.items()}})
        del idx

    del model, emb_cache
    gc.collect(); torch.cuda.empty_cache()

# Summary
print("\n" + "="*70)
print("ALL RESULTS")
print("="*70)
df = pd.DataFrame(results_log).sort_values('FinalScore', ascending=False)
print(df[['name', 'SR@1', 'VR@1', 'AvgSR', 'AvgVR', 'FinalScore']].to_string(index=False))
best = df.iloc[0]
print(f"\nBEST: {best['name']} FinalScore={best['FinalScore']:.4f}")
df.to_csv(f'{WORK}/eval_finetuned_summary.csv', index=False)

# Generate submission for the best config
best_model_name = best['model']
best_model_path = MODELS[best_model_name]
print(f"\nGenerating submission for best config: {best['name']}")

model = SentenceTransformer(best_model_path, device=DEVICE, trust_remote_code=True)
chunks = build_chunks(chunk_configs[best['chunks']], transcripts)
texts = [ch['text'] for ch in chunks]
emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True).astype('float32')
idx = faiss.IndexFlatIP(emb.shape[1])
idx.add(emb)

test_r = []
for _, row in tqdm(test.iterrows(), total=len(test), desc='Best submission'):
    q = str(row['question']).lower().strip()
    qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    hits = retrieve(qv, idx, chunks, int(best['top_k']), 5, float(best['gap']), float(best['shrink']))
    test_r.append({'query_id': row['query_id'], 'hits': hits})

rows = []
for r in test_r:
    d = {'query_id': r['query_id']}
    for rk in range(1, 6):
        if rk <= len(r['hits']):
            h = r['hits'][rk-1]
            d[f'video_file_{rk}'] = h2f.get(h['video_hash'], h['video_hash'])
            d[f'start_{rk}'] = round(h['start'], 1)
            d[f'end_{rk}'] = round(h['end'], 1)
        else:
            d[f'video_file_{rk}'], d[f'start_{rk}'], d[f'end_{rk}'] = '', 0.0, 0.0
    rows.append(d)
cols = ['query_id']
for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
sub_path = f'{WORK}/submission_best_finetuned.csv'
pd.DataFrame(rows, columns=cols).to_csv(sub_path, index=False)
print(f"Best submission saved: {sub_path}")
print("Done!")
