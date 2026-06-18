#!/usr/bin/env python3
"""V3: Fine-tune around best config (bge_m3 + small+medium chunks + shrink).
Also try: higher shrink values, different top_k, cross-model ensembling.
"""

import pickle, re, os, shutil, json, numpy as np, pandas as pd, faiss, torch, time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)

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

# ── Load data ───────────────────────────────────────────────────
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

# ── Pre-build all chunk sets we'll need ─────────────────────────
chunk_configs = {
    'sm': [(30.0, 15.0), (60.0, 30.0)],
    'sm_dense': [(30.0, 10.0), (60.0, 20.0)],
    's': [(30.0, 15.0)],
    's_dense': [(30.0, 10.0)],
    'sml': [(30.0, 15.0), (60.0, 30.0), (90.0, 45.0)],
}

chunk_sets = {}
for name, cfg in chunk_configs.items():
    chunk_sets[name] = build_chunks(cfg, transcripts)
    print(f"Chunks '{name}': {len(chunk_sets[name])}")

# ── Load model and encode ───────────────────────────────────────
print("Loading BGE-M3...")
model = SentenceTransformer('BAAI/bge-m3', device=DEVICE, trust_remote_code=True)

embeddings_cache = {}
for name, chunks in chunk_sets.items():
    print(f"Encoding '{name}' ({len(chunks)} chunks)...")
    texts = [ch['text'] for ch in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    embeddings_cache[name] = emb

# ── Experiments ─────────────────────────────────────────────────
EXPERIMENTS = [
    # Vary shrink around best
    {'name': 'sm_shrink20', 'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.2},
    {'name': 'sm_shrink30', 'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.3},
    {'name': 'sm_shrink40', 'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.4},
    {'name': 'sm_shrink50', 'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.5},
    {'name': 'sm_shrink60', 'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.6},
    {'name': 'sm_shrink70', 'chunks': 'sm', 'top_k': 100, 'gap': 10, 'shrink': 0.7},
    # Vary gap
    {'name': 'sm_s30_gap5', 'chunks': 'sm', 'top_k': 100, 'gap': 5, 'shrink': 0.3},
    {'name': 'sm_s30_gap15', 'chunks': 'sm', 'top_k': 100, 'gap': 15, 'shrink': 0.3},
    {'name': 'sm_s30_gap20', 'chunks': 'sm', 'top_k': 100, 'gap': 20, 'shrink': 0.3},
    # Vary top_k
    {'name': 'sm_s30_topk50', 'chunks': 'sm', 'top_k': 50, 'gap': 10, 'shrink': 0.3},
    {'name': 'sm_s30_topk200', 'chunks': 'sm', 'top_k': 200, 'gap': 10, 'shrink': 0.3},
    # Dense stepping
    {'name': 'sm_dense_s30', 'chunks': 'sm_dense', 'top_k': 100, 'gap': 10, 'shrink': 0.3},
    # Small only with shrink
    {'name': 's_s30', 'chunks': 's', 'top_k': 100, 'gap': 10, 'shrink': 0.3},
    {'name': 's_s50', 'chunks': 's', 'top_k': 100, 'gap': 10, 'shrink': 0.5},
    {'name': 's_dense_s30', 'chunks': 's_dense', 'top_k': 100, 'gap': 10, 'shrink': 0.3},
    # 3 scales
    {'name': 'sml_s30', 'chunks': 'sml', 'top_k': 100, 'gap': 10, 'shrink': 0.3},
    {'name': 'sml_s50', 'chunks': 'sml', 'top_k': 100, 'gap': 10, 'shrink': 0.5},
    # Best combos with higher top_k
    {'name': 'sm_s50_topk200', 'chunks': 'sm', 'top_k': 200, 'gap': 10, 'shrink': 0.5},
    {'name': 'sm_s40_gap5', 'chunks': 'sm', 'top_k': 100, 'gap': 5, 'shrink': 0.4},
]

results_log = []

for exp in EXPERIMENTS:
    print(f"\n--- {exp['name']} ---")
    t0 = time.time()
    chunks = chunk_sets[exp['chunks']]
    emb = embeddings_cache[exp['chunks']]
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    tr = []
    for _, row in tq.iterrows():
        q = row['question_en'].lower().strip()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qv, idx, chunks, exp['top_k'], 5, exp['gap'], exp['shrink'])
        tr.append({'query_id': row['question_id'], 'hits': hits})

    m = evaluate(gt, tr)
    elapsed = time.time() - t0
    print(f"  SR@1={m['SR@1']:.4f} SR@5={m['SR@5']:.4f} VR@1={m['VR@1']:.4f} VR@5={m['VR@5']:.4f}")
    print(f"  AvgSR={m['AvgSR']:.4f} AvgVR={m['AvgVR']:.4f} FinalScore={m['FinalScore']:.4f} ({elapsed:.0f}s)")

    results_log.append({'name': exp['name'], **exp, 'time_s': round(elapsed,1),
                        **{k: round(v,4) for k,v in m.items()}})

    # Generate submission for best candidates (FinalScore > 0.36)
    if m['FinalScore'] > 0.36:
        print(f"  Generating submission (score > 0.36)...")
        test_r = []
        for _, row in test.iterrows():
            q = str(row['question']).lower().strip()
            qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
            hits = retrieve(qv, idx, chunks, exp['top_k'], 5, exp['gap'], exp['shrink'])
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
        pd.DataFrame(rows, columns=cols).to_csv(f'{WORK}/submission_{exp["name"]}.csv', index=False)

    del idx

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("V3 EXPERIMENT SUMMARY")
print(f"{'='*60}")
df = pd.DataFrame(results_log).sort_values('FinalScore', ascending=False)
print(df[['name', 'SR@1', 'SR@5', 'VR@1', 'VR@5', 'AvgSR', 'AvgVR', 'FinalScore']].to_string(index=False))
best = df.iloc[0]
print(f"\nBEST: {best['name']} FinalScore={best['FinalScore']:.4f}")
df.to_csv(f'{WORK}/experiment_v3_summary.csv', index=False)
print("Done!")
