#!/usr/bin/env python3
"""submission.ipynb adapted for Vast. Only change: bge -> e5-ft, paths -> /root/data"""
import os, pickle, re, json
import numpy as np, pandas as pd, faiss, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORKDIR = '/root/output'
CACHE_DIR = os.path.join(WORKDIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
TOP_K_FAISS = 100
TOP_K_RERANK = 10
TOP_K_FINAL = 5
MAX_PER_VIDEO = 2
OVERLAP_THRESHOLD = 0.5
USE_CLIP = False
print(f'Device: {DEVICE}')

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\.\w]*$', str(path))
    return m.group(1) if m else None

def merge_and_shrink(candidates, gap=10.0, shrink=0.7):
    by_video = {}
    for c in candidates:
        by_video.setdefault(c['video_hash'], []).append(c)
    merged = []
    for vh, chks in by_video.items():
        chks = sorted(chks, key=lambda x: x['start'])
        cur, best = chks[0].copy(), chks[0].copy()
        for nxt in chks[1:]:
            if nxt['start'] <= cur['end'] + gap:
                cur['end'] = max(cur['end'], nxt['end'])
                if nxt['score'] > cur['score']:
                    cur['score'] = nxt['score']; best = nxt.copy()
            else:
                bc = (best['start'] + best['end']) / 2
                bh = (best['end'] - best['start']) / 2
                ch = (cur['end'] - cur['start']) / 2
                nh = ch * (1 - shrink) + bh * shrink
                cur['start'] = max(cur['start'], bc - nh)
                cur['end'] = min(cur['end'], bc + nh)
                merged.append(cur)
                cur, best = nxt.copy(), nxt.copy()
        bc = (best['start'] + best['end']) / 2
        bh = (best['end'] - best['start']) / 2
        ch = (cur['end'] - cur['start']) / 2
        nh = ch * (1 - shrink) + bh * shrink
        cur['start'] = max(cur['start'], bc - nh)
        cur['end'] = min(cur['end'], bc + nh)
        merged.append(cur)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

def make_chunks(segments, window, step):
    chunks = []
    if not segments: return chunks
    t = segments[0]['start']
    while t < segments[-1]['end']:
        idxs = [i for i, s in enumerate(segments) if s['end'] > t and s['start'] < t + window]
        if idxs:
            i0, i1 = idxs[0], idxs[-1]
            ws = segments[i0:i1+1]
            chunks.append({
                'start': ws[0]['start'], 'end': ws[-1]['end'],
                'text': ' '.join(s['text'].lower().strip() for s in ws),
                'seg_start_idx': i0, 'seg_end_idx': i1,
            })
        t += step
    return chunks

with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)
h2f = {}
for p in pd.read_csv(f'{BASE}/video_files.csv')['video_path']:
    h = extract_hash(p)
    if h: h2f[h] = re.sub(r'\.\w+$', '', p.split('/')[-1])
transcripts_by_hash = {}
for key, segs in transcripts.items():
    vh = extract_hash(key)
    if vh: transcripts_by_hash[vh] = segs
test = pd.read_csv(f'{BASE}/test/test.csv')
print(f'Test: {len(test)} queries, Videos: {len(h2f)}')

all_chunks = []
for key, segs in tqdm(transcripts.items()):
    vh = extract_hash(key)
    if not vh or vh in SKIP_HASHES: continue
    for w, s in [(30.0, 15.0), (60.0, 30.0)]:
        for ch in make_chunks(segs, w, s):
            ch['video_hash'] = vh
            all_chunks.append(ch)
print(f'Chunks: {len(all_chunks)}')

def expand_query(query, max_expansions=6):
    q = str(query).lower().strip()
    if not q: return [q]
    phrase_map = {'refrigerator':['fridge'],'open the':['look inside the'],'open':['look inside','unlock'],'take':['remove','grab'],'pick up':['lift','grab'],'put down':['place','set down'],'turn on':['switch on','power on'],'turn off':['switch off','power off']}
    token_map = {'refrigerator':['fridge'],'cupboard':['cabinet'],'bottle':['flask'],'sofa':['couch'],'television':['tv'],'shelf':['rack'],'backpack':['bag'],'laptop':['notebook']}
    expansions = {q}
    for k, reps in phrase_map.items():
        if k in q:
            for r in reps: expansions.add(q.replace(k, r))
    tokens = q.split()
    for i, tok in enumerate(tokens):
        if tok in token_map:
            for r in token_map[tok]:
                nt = tokens.copy(); nt[i] = r; expansions.add(' '.join(nt))
    if q.startswith('open '): expansions.add(q.replace('open ', 'look inside ', 1))
    if q.startswith('close '): expansions.add(q.replace('close ', 'shut ', 1))
    out = list(expansions)
    if q in out: out.remove(q); out = [q] + out
    return out[:max_expansions]

# === ONLY CHANGE: e5-ft instead of bge-m3-ft ===
model = SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2', device=DEVICE, trust_remote_code=True)

emb = model.encode([ch['text'] for ch in all_chunks], batch_size=64,
                    show_progress_bar=True, normalize_embeddings=True,
                    convert_to_numpy=True).astype('float32')
print(f'Chunk embeddings: {emb.shape}')

segment_emb_cache = {}
def get_segment_embeddings(video_hash):
    if video_hash in segment_emb_cache: return segment_emb_cache[video_hash]
    segs = transcripts_by_hash[video_hash]
    seg_emb = model.encode([s['text'].lower().strip() for s in segs], batch_size=64,
                            show_progress_bar=False, normalize_embeddings=True,
                            convert_to_numpy=True).astype('float32')
    segment_emb_cache[video_hash] = seg_emb
    return seg_emb

dim = emb.shape[1]
try:
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
except TypeError:
    index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = max(100, TOP_K_FAISS * 2)
index.add(emb)
print(f'Index: {index.ntotal} vectors, dim={dim}')

def encode_query(query):
    expanded = expand_query(query)
    q_emb = model.encode(expanded, batch_size=16, show_progress_bar=False,
                          normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    qv = q_emb.mean(axis=0, keepdims=True)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
    return qv, expanded

query_cache = {}
def retrieve_candidates(query):
    q = str(query).lower().strip()
    if q in query_cache: qv = query_cache[q]
    else: qv, _ = encode_query(q); query_cache[q] = qv
    scores, ids = index.search(qv, TOP_K_FAISS)
    cands = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1: continue
        ch = all_chunks[i]
        cands.append({'video_hash':ch['video_hash'],'start':ch['start'],'end':ch['end'],
                      'text':ch['text'],'seg_start_idx':ch['seg_start_idx'],
                      'seg_end_idx':ch['seg_end_idx'],'faiss_score':float(s)})
    return qv, cands

reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=DEVICE)

def rerank(query, candidates, batch_size=32, top_k=TOP_K_RERANK):
    if not candidates: return []
    pairs = [(query, c['text']) for c in candidates]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c['rerank_score'] = float(s); c['score'] = float(s)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]

def refine_timestamp(query_emb, cand):
    vh = cand['video_hash']
    segs = transcripts_by_hash.get(vh)
    if not segs: return cand
    i0 = max(0, cand['seg_start_idx'])
    i1 = min(len(segs) - 1, cand['seg_end_idx'])
    seg_emb = get_segment_embeddings(vh)
    seg_slice = seg_emb[i0:i1+1]
    if seg_slice.size == 0: return cand
    sims = seg_slice @ query_emb[0]
    best_idx = int(np.argmax(sims))
    best_seg = segs[i0 + best_idx]
    cand['start'] = float(best_seg['start'])
    cand['end'] = float(best_seg['end'])
    cand['segment_score'] = float(sims[best_idx])
    return cand

def overlap_ratio(a, b):
    inter = max(0.0, min(a['end'], b['end']) - max(a['start'], b['start']))
    if inter <= 0: return 0.0
    return inter / min(max(1e-6, a['end']-a['start']), max(1e-6, b['end']-b['start']))

def diversity_filter(candidates, top_k=TOP_K_FINAL, max_per_video=MAX_PER_VIDEO, overlap_thr=OVERLAP_THRESHOLD):
    selected = []; per_video = {}
    for c in sorted(candidates, key=lambda x: x['score'], reverse=True):
        vh = c['video_hash']
        if per_video.get(vh, 0) >= max_per_video: continue
        if any(s['video_hash'] == vh and overlap_ratio(c, s) > overlap_thr for s in selected): continue
        selected.append(c); per_video[vh] = per_video.get(vh, 0) + 1
        if len(selected) >= top_k: break
    return selected

print('\nRunning pipeline...')
results = []
for _, row in tqdm(test.iterrows(), total=len(test)):
    query = str(row['question']).lower().strip()
    qv, cands = retrieve_candidates(query)
    reranked = rerank(query, cands, top_k=TOP_K_RERANK)
    refined = [refine_timestamp(qv, c.copy()) for c in reranked]
    final_hits = diversity_filter(refined, top_k=TOP_K_FINAL)
    results.append({'query_id': row['query_id'], 'hits': final_hits})

fallback = list(h2f.values())[0]
rows = []
for r in results:
    d = {'query_id': r['query_id']}
    for rk in range(1, 6):
        if rk <= len(r['hits']):
            h = r['hits'][rk-1]
            d[f'video_file_{rk}'] = h2f.get(h['video_hash'], fallback)
            d[f'start_{rk}'] = round(h['start'], 1)
            d[f'end_{rk}'] = round(h['end'], 1)
        else:
            d[f'video_file_{rk}'] = fallback
            d[f'start_{rk}'] = 0.0; d[f'end_{rk}'] = 1.0
    rows.append(d)
cols = ['query_id']
for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
sub = pd.DataFrame(rows, columns=cols)
sub.to_csv(f'{WORKDIR}/submission_nb_e5ft.csv', index=False)
print(f'Saved. Shape: {sub.shape}')
nulls = sub.isnull().sum().sum() + (sub == '').sum().sum()
print(f'Nulls: {nulls}')
print('Done!')
