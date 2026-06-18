#!/usr/bin/env python3
"""Extra experiments: try more extreme shrink values and combos for the best model."""

import pickle, re, os, numpy as np, pandas as pd, faiss, torch, gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None
def clean_text(text): return text.lower().strip()
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
def build_chunks(cfg, tr):
    c = []
    for key, segs in tr.items():
        vh = extract_hash(key)
        if vh is None or vh in SKIP_HASHES: continue
        for w, s in cfg:
            for ch in make_chunks(segs, w, s):
                ch['video_hash'] = vh; c.append(ch)
    return c
def merge_chunks_v2(candidates, gap, shrink_factor=0.0):
    by_video = {}
    for c in candidates: by_video.setdefault(c['video_hash'], []).append(c)
    merged = []
    for vh, chks in by_video.items():
        chks = sorted(chks, key=lambda x: x['start'])
        current, best = chks[0].copy(), chks[0].copy()
        for nxt in chks[1:]:
            if nxt['start'] <= current['end'] + gap:
                current['end'] = max(current['end'], nxt['end'])
                if nxt['score'] > current['score']:
                    current['score'] = nxt['score']; best = nxt.copy()
            else:
                if shrink_factor > 0:
                    bc=(best['start']+best['end'])/2; bh=(best['end']-best['start'])/2
                    ch=(current['end']-current['start'])/2; nh=ch*(1-shrink_factor)+bh*shrink_factor
                    current['start']=max(current['start'],bc-nh); current['end']=min(current['end'],bc+nh)
                merged.append(current); current, best = nxt.copy(), nxt.copy()
        if shrink_factor > 0:
            bc=(best['start']+best['end'])/2; bh=(best['end']-best['start'])/2
            ch=(current['end']-current['start'])/2; nh=ch*(1-shrink_factor)+bh*shrink_factor
            current['start']=max(current['start'],bc-nh); current['end']=min(current['end'],bc+nh)
        merged.append(current)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged
def retrieve(qvec, index, chunks, top_k=100, top_n=5, gap=10.0, shrink=0.0):
    scores, indices = index.search(qvec, top_k)
    cands = [{'video_hash': chunks[i]['video_hash'], 'start': chunks[i]['start'],
              'end': chunks[i]['end'], 'score': float(s)}
             for s, i in zip(scores[0], indices[0]) if i != -1]
    return merge_chunks_v2(cands, gap, shrink)[:top_n]
def iou(ps,pe,gs,ge):
    inter=max(0,min(pe,ge)-max(ps,gs)); union=max(pe,ge)-min(ps,gs)
    return inter/union if union>0 else 0.0
def evaluate(gt, results, ks=(1,3,5)):
    sr,vr = {k:[] for k in ks},{k:[] for k in ks}
    for r in results:
        if r['query_id'] not in gt: continue
        gi=gt[r['query_id']]
        for k in ks:
            th=r['hits'][:k]
            vr[k].append(int(bool({h['video_hash'] for h in th}&{g['video_hash'] for g in gi})))
            sh=0
            for h in th:
                for g in gi:
                    if h['video_hash']==g['video_hash'] and iou(h['start'],h['end'],g['start'],g['end'])>=0.5: sh=1; break
                if sh: break
            sr[k].append(sh)
    m={}
    for k in ks: m[f'SR@{k}'],m[f'VR@{k}']=np.mean(sr[k]),np.mean(vr[k])
    m['AvgSR']=np.mean([m[f'SR@{k}'] for k in ks])
    m['AvgVR']=np.mean([m[f'VR@{k}'] for k in ks])
    m['FinalScore']=(m['AvgSR']+m['AvgVR'])/2
    return m

print("Loading data...")
with open(f'{BASE}/transcripts.pkl','rb') as f: transcripts=pickle.load(f)
train=pd.read_csv(f'{BASE}/train/train_qa.csv')
gt={}
for _,row in train.iterrows():
    qid,vh=row['question_id'],extract_hash(row['video_file'])
    gt.setdefault(qid,[]).append({'video_hash':vh,'start':row['start'],'end':row['end']})
tq=train[['question_id','question_en']].drop_duplicates('question_id')
vf=pd.read_csv(f'{BASE}/video_files.csv')
h2f={}
for p in vf['video_path']:
    h=extract_hash(p); fn=re.sub(r'\.\w+$','',p.split('/')[-1])
    if h: h2f[h]=fn
test=pd.read_csv(f'{BASE}/test/test.csv')

# Load the best model (ft_e5_large)
print("Loading ft_e5_large...")
model = SentenceTransformer('/root/output/finetuned_e5_large', device=DEVICE, trust_remote_code=True)

# Build chunks
cfgs = {
    'sm': [(30.0,15.0),(60.0,30.0)],
    'sml': [(30.0,15.0),(60.0,30.0),(90.0,45.0)],
}
chunk_cache, emb_cache = {}, {}
for cn, cfg in cfgs.items():
    chunks = build_chunks(cfg, transcripts)
    chunk_cache[cn] = chunks
    texts = [ch['text'] for ch in chunks]
    print(f"Encoding {cn} ({len(texts)} chunks)...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    emb_cache[cn] = emb

# Encode queries
print("Encoding queries...")
q_texts = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
q_ids = [row['question_id'] for _, row in tq.iterrows()]
q_embs = model.encode(q_texts, batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype('float32')

EXPERIMENTS = [
    # Higher shrink values for sm
    {'chunks':'sm','top_k':100,'gap':10,'shrink':0.95},
    {'chunks':'sm','top_k':100,'gap':10,'shrink':1.0},
    # sml with various shrink
    {'chunks':'sml','top_k':100,'gap':10,'shrink':0.8},
    {'chunks':'sml','top_k':100,'gap':10,'shrink':0.9},
    {'chunks':'sml','top_k':100,'gap':10,'shrink':0.95},
    # sml with different top_k
    {'chunks':'sml','top_k':50,'gap':10,'shrink':0.9},
    {'chunks':'sml','top_k':50,'gap':10,'shrink':0.7},
    # sm with top_k=50 and higher shrink
    {'chunks':'sm','top_k':50,'gap':10,'shrink':0.9},
    {'chunks':'sm','top_k':50,'gap':10,'shrink':0.8},
    {'chunks':'sm','top_k':50,'gap':10,'shrink':0.95},
    # sm with top_k=30
    {'chunks':'sm','top_k':30,'gap':10,'shrink':0.9},
    {'chunks':'sm','top_k':30,'gap':10,'shrink':0.95},
]

results_log = []
best_score = 0
best_exp = None

for exp in EXPERIMENTS:
    cn = exp['chunks']
    chunks = chunk_cache[cn]
    emb = emb_cache[cn]
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    tr = []
    for i, qid in enumerate(q_ids):
        hits = retrieve(q_embs[i:i+1], idx, chunks, exp['top_k'], 5, exp['gap'], exp['shrink'])
        tr.append({'query_id': qid, 'hits': hits})
    m = evaluate(gt, tr)
    name = f"e5ft_{cn}_k{exp['top_k']}_g{exp['gap']}_s{int(exp['shrink']*100)}"
    print(f"{name}: SR@1={m['SR@1']:.4f} VR@1={m['VR@1']:.4f} AvgSR={m['AvgSR']:.4f} AvgVR={m['AvgVR']:.4f} FS={m['FinalScore']:.4f}")
    results_log.append({'name': name, **exp, **{k: round(v,4) for k,v in m.items()}})
    if m['FinalScore'] > best_score:
        best_score = m['FinalScore']
        best_exp = exp.copy()
        best_exp['name'] = name
    del idx

print(f"\nBest extra experiment: {best_exp['name']} FS={best_score:.4f}")

# Generate submission for best
if best_score > 0.5191:  # Only if better than previous best
    print(f"Generating improved submission...")
    cn = best_exp['chunks']
    chunks = chunk_cache[cn]
    emb = emb_cache[cn]
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    test_r = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc='Submission'):
        q = str(row['question']).lower().strip()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qv, idx, chunks, best_exp['top_k'], 5, best_exp['gap'], best_exp['shrink'])
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

pd.DataFrame(results_log).to_csv(f'{WORK}/eval_extra_summary.csv', index=False)
print("Done!")
