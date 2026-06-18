#!/usr/bin/env python3
"""Test dense chunk overlap for better boundary alignment."""
import pickle, re, os, numpy as np, pandas as pd, faiss, torch, gc, time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = "/root/data/video-rag"
WORK = "/root/output"
DEVICE = "cuda"
SKIP_HASHES = {"7d49c038"}

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def make_chunks(segments, window, step):
    if not segments: return []
    chunks, t = [], segments[0]['start']
    while t < segments[-1]['end']:
        ws = [s for s in segments if s['end'] > t and s['start'] < t + window]
        if ws:
            chunks.append({'start': ws[0]['start'], 'end': ws[-1]['end'],
                          'text': ' '.join(s['text'].lower().strip() for s in ws)})
        t += step
    return chunks

def merge(cands, gap, shrink):
    by_video = {}
    for c in cands:
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
                if shrink > 0:
                    bc=(best['start']+best['end'])/2; bh=(best['end']-best['start'])/2
                    ch=(cur['end']-cur['start'])/2; nh=ch*(1-shrink)+bh*shrink
                    cur['start']=max(cur['start'],bc-nh); cur['end']=min(cur['end'],bc+nh)
                merged.append(cur)
                cur, best = nxt.copy(), nxt.copy()
        if shrink > 0:
            bc=(best['start']+best['end'])/2; bh=(best['end']-best['start'])/2
            ch=(cur['end']-cur['start'])/2; nh=ch*(1-shrink)+bh*shrink
            cur['start']=max(cur['start'],bc-nh); cur['end']=min(cur['end'],bc+nh)
        merged.append(cur)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

def iou(ps,pe,gs,ge):
    i=max(0,min(pe,ge)-max(ps,gs)); u=max(pe,ge)-min(ps,gs)
    return i/u if u>0 else 0

def evaluate(gt, results):
    sr,vr={k:[] for k in (1,3,5)},{k:[] for k in (1,3,5)}
    for r in results:
        if r['query_id'] not in gt: continue
        gi=gt[r['query_id']]
        for k in (1,3,5):
            th=r['hits'][:k]
            vr[k].append(int(bool({h['video_hash'] for h in th}&{g['video_hash'] for g in gi})))
            sh=0
            for h in th:
                for g in gi:
                    if h['video_hash']==g['video_hash'] and iou(h['start'],h['end'],g['start'],g['end'])>=0.5:
                        sh=1;break
                if sh:break
            sr[k].append(sh)
    m={}
    for k in (1,3,5):
        m[f'SR@{k}'],m[f'VR@{k}']=np.mean(sr[k]),np.mean(vr[k])
    m['AvgSR']=np.mean([m[f'SR@{k}'] for k in (1,3,5)])
    m['AvgVR']=np.mean([m[f'VR@{k}'] for k in (1,3,5)])
    m['FinalScore']=(m['AvgSR']+m['AvgVR'])/2
    return m

# Load
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)
th = {}
for key, segs in transcripts.items():
    vh = extract_hash(key)
    if vh and vh not in SKIP_HASHES: th[vh] = segs

train = pd.read_csv(f'{BASE}/train/train_qa.csv')
test = pd.read_csv(f'{BASE}/test/test.csv')
vf = pd.read_csv(f'{BASE}/video_files.csv')
h2f = {}
for p in vf['video_path']:
    h = extract_hash(p); fn = re.sub(r'\.\w+$', '', p.split('/')[-1])
    if h: h2f[h] = fn

gt = {}
for _, row in train.iterrows():
    qid, vh = row['question_id'], extract_hash(row['video_file'])
    gt.setdefault(qid, []).append({'video_hash': vh, 'start': row['start'], 'end': row['end']})
tq = train[['question_id', 'question_en']].drop_duplicates('question_id')

print("Loading model...")
model = SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2', device=DEVICE, trust_remote_code=True)

queries = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
qids = [row['question_id'] for _, row in tq.iterrows()]
q_emb = model.encode(queries, batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True).astype('float32')

CONFIGS = [
    {"name": "baseline_50pct_overlap", "chunks": [(30,15),(60,30),(90,45)], "shrink": 0.95},
    {"name": "dense_83pct_30_60_90", "chunks": [(30,5),(60,10),(90,15)], "shrink": 0.95},
    {"name": "dense_83pct_20_40_60", "chunks": [(20,5),(40,10),(60,15)], "shrink": 0.95},
    {"name": "dense_83pct_15_30_60", "chunks": [(15,5),(30,10),(60,15)], "shrink": 0.95},
    {"name": "dense_90pct_30_s3", "chunks": [(30,3)], "shrink": 0.95},
    {"name": "dense_90pct_60_s6", "chunks": [(60,6)], "shrink": 0.95},
    {"name": "dense_83pct_30_60_s5_s10", "chunks": [(30,5),(60,10)], "shrink": 0.95},
    {"name": "dense_83pct_30_60_90_shrink80", "chunks": [(30,5),(60,10),(90,15)], "shrink": 0.80},
    {"name": "dense_83pct_30_60_90_shrink100", "chunks": [(30,5),(60,10),(90,15)], "shrink": 1.0},
    {"name": "dense_83pct_30_60_90_gap5", "chunks": [(30,5),(60,10),(90,15)], "shrink": 0.95, "gap": 5},
    {"name": "dense_83pct_30_60_90_gap0", "chunks": [(30,5),(60,10),(90,15)], "shrink": 0.95, "gap": 0},
]

best_score = 0
best_cfg = None

for cfg in CONFIGS:
    name = cfg["name"]
    gap = cfg.get("gap", 10)
    shrink = cfg["shrink"]
    t0 = time.time()

    chunks = []
    for vh, segs in th.items():
        for w, s in cfg["chunks"]:
            for ch in make_chunks(segs, w, s):
                ch['video_hash'] = vh
                chunks.append(ch)

    emb = model.encode([c['text'] for c in chunks], batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    scores, indices = idx.search(q_emb, 100)
    results = []
    for i in range(len(qids)):
        cands = [{'video_hash': chunks[j]['video_hash'], 'start': chunks[j]['start'],
                  'end': chunks[j]['end'], 'score': float(s)}
                 for s, j in zip(scores[i], indices[i]) if j != -1]
        hits = merge(cands, gap, shrink)[:5]
        results.append({'query_id': qids[i], 'hits': hits})

    m = evaluate(gt, results)
    elapsed = time.time() - t0
    fs = m['FinalScore']
    n_chunks = len(chunks)
    print(f"{name}: chunks={n_chunks} SR@1={m['SR@1']:.4f} SR@5={m['SR@5']:.4f} VR@1={m['VR@1']:.4f} VR@5={m['VR@5']:.4f} AvgSR={m['AvgSR']:.4f} AvgVR={m['AvgVR']:.4f} FS={fs:.4f} ({elapsed:.0f}s)")

    if fs > best_score:
        best_score = fs
        best_cfg = cfg
        print(f"  *** NEW BEST ***")

    del emb, idx, chunks
    gc.collect(); torch.cuda.empty_cache()

print(f"\nBEST: {best_cfg['name']} => FinalScore={best_score:.4f}")

# Generate test submission
print(f"\nGenerating test submission...")
chunks = []
for vh, segs in th.items():
    for w, s in best_cfg["chunks"]:
        for ch in make_chunks(segs, w, s):
            ch['video_hash'] = vh
            chunks.append(ch)
emb = model.encode([c['text'] for c in chunks], batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True).astype('float32')
idx = faiss.IndexFlatIP(emb.shape[1])
idx.add(emb)

tq_test = [str(row['question']).lower().strip() for _, row in test.iterrows()]
tq_ids = [row['query_id'] for _, row in test.iterrows()]
qt_emb = model.encode(tq_test, batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True).astype('float32')
ts, ti = idx.search(qt_emb, 100)

fallback = list(h2f.values())[0]
rows = []
for i in range(len(tq_ids)):
    cands = [{'video_hash': chunks[j]['video_hash'], 'start': chunks[j]['start'],
              'end': chunks[j]['end'], 'score': float(s)}
             for s, j in zip(ts[i], ti[i]) if j != -1]
    hits = merge(cands, best_cfg.get("gap", 10), best_cfg["shrink"])[:5]
    d = {'query_id': tq_ids[i]}
    for rk in range(1, 6):
        if rk <= len(hits):
            h = hits[rk-1]
            d[f'video_file_{rk}'] = h2f.get(h['video_hash'], fallback)
            d[f'start_{rk}'] = round(h['start'], 1)
            d[f'end_{rk}'] = round(h['end'], 1)
        else:
            d[f'video_file_{rk}'] = fallback
            d[f'start_{rk}'] = 0.0
            d[f'end_{rk}'] = 1.0
    rows.append(d)

cols = ['query_id']
for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
sub = pd.DataFrame(rows, columns=cols)
sub.to_csv(f'{WORK}/submission_dense_best.csv', index=False)
print(f"Saved: {WORK}/submission_dense_best.csv")
print("Done!")
