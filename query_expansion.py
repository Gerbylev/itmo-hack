#!/usr/bin/env python3
"""Query expansion: search -> take top chunk text -> re-search with expanded query."""
import pickle, re, numpy as np, pandas as pd, faiss, torch, time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE="/root/data/video-rag"; SKIP={"7d49c038"}

def xh(p):
    m=re.search(r'_([a-f0-9]+)[\.\w]*$',str(p))
    return m.group(1) if m else None

def mkc(segs,w,s):
    ch,t=[],segs[0]['start']
    while t<segs[-1]['end']:
        ws=[x for x in segs if x['end']>t and x['start']<t+w]
        if ws: ch.append({'start':ws[0]['start'],'end':ws[-1]['end'],'text':' '.join(x['text'].lower().strip() for x in ws)})
        t+=s
    return ch

def merge(c,gap,sh):
    bv={}
    for x in c: bv.setdefault(x['video_hash'],[]).append(x)
    m=[]
    for vh,ck in bv.items():
        ck=sorted(ck,key=lambda x:x['start']); cur,best=ck[0].copy(),ck[0].copy()
        for nxt in ck[1:]:
            if nxt['start']<=cur['end']+gap:
                cur['end']=max(cur['end'],nxt['end'])
                if nxt['score']>cur['score']: cur['score']=nxt['score'];best=nxt.copy()
            else:
                if sh>0:
                    bc=(best['start']+best['end'])/2;bh=(best['end']-best['start'])/2
                    ch2=(cur['end']-cur['start'])/2;nh=ch2*(1-sh)+bh*sh
                    cur['start']=max(cur['start'],bc-nh);cur['end']=min(cur['end'],bc+nh)
                m.append(cur);cur,best=nxt.copy(),nxt.copy()
        if sh>0:
            bc=(best['start']+best['end'])/2;bh=(best['end']-best['start'])/2
            ch2=(cur['end']-cur['start'])/2;nh=ch2*(1-sh)+bh*sh
            cur['start']=max(cur['start'],bc-nh);cur['end']=min(cur['end'],bc+nh)
        m.append(cur)
    m.sort(key=lambda x:x['score'],reverse=True);return m

def iou(ps,pe,gs,ge):
    i=max(0,min(pe,ge)-max(ps,gs));u=max(pe,ge)-min(ps,gs)
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
with open(f'{BASE}/transcripts.pkl','rb') as f: tr=pickle.load(f)
th={}
for k,s in tr.items():
    vh=xh(k)
    if vh and vh not in SKIP: th[vh]=s

train=pd.read_csv(f'{BASE}/train/train_qa.csv')
test=pd.read_csv(f'{BASE}/test/test.csv')
vf=pd.read_csv(f'{BASE}/video_files.csv')
h2f={}
for p in vf['video_path']:
    h=xh(p);fn=re.sub(r'\.\w+$','',p.split('/')[-1])
    if h:h2f[h]=fn

gt={}
for _,row in train.iterrows():
    qid,vh=row['question_id'],xh(row['video_file'])
    gt.setdefault(qid,[]).append({'video_hash':vh,'start':row['start'],'end':row['end']})
tq=train[['question_id','question_en']].drop_duplicates('question_id')

print('Loading model...')
model=SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2',device='cuda',trust_remote_code=True)

# Fast config: 38K chunks
chunks=[]
for vh,segs in th.items():
    for w,s in [(30,15),(60,30),(90,45)]:
        for ch in mkc(segs,w,s):
            ch['video_hash']=vh;chunks.append(ch)
print(f'Chunks: {len(chunks)}')

emb=model.encode([c['text'] for c in chunks],batch_size=64,show_progress_bar=True,
                 normalize_embeddings=True,convert_to_numpy=True).astype('float32')
idx=faiss.IndexFlatIP(emb.shape[1]);idx.add(emb)

# ── Baseline (no expansion) ────────────────────────────────
print('\n=== Baseline (no expansion) ===')
queries=[row['question_en'].lower().strip() for _,row in tq.iterrows()]
qids=[row['question_id'] for _,row in tq.iterrows()]
q_emb=model.encode(queries,batch_size=64,normalize_embeddings=True,
                   convert_to_numpy=True).astype('float32')
sc,ix=idx.search(q_emb,100)

results_base=[]
for i in range(len(qids)):
    ca=[{'video_hash':chunks[j]['video_hash'],'start':chunks[j]['start'],'end':chunks[j]['end'],'score':float(s)}
        for s,j in zip(sc[i],ix[i]) if j!=-1]
    hits=merge(ca,10,0.95)[:5]
    results_base.append({'query_id':qids[i],'hits':hits})
m=evaluate(gt,results_base)
print(f"Baseline: AvgSR={m['AvgSR']:.4f} AvgVR={m['AvgVR']:.4f} FS={m['FinalScore']:.4f}")

# ── Query Expansion variants ───────────────────────────────
def expand_and_search(queries, qids, idx, chunks, model, method, alpha=0.5):
    """
    method: 'concat' - append top chunk text to query
            'interpolate' - interpolate query and top chunk embeddings
            'top3_concat' - append text from top 3 chunks
    """
    q_emb = model.encode(queries, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True).astype('float32')

    # First pass
    sc1, ix1 = idx.search(q_emb, 100)

    if method == 'interpolate':
        # Interpolate query embedding with top-1 chunk embedding
        expanded_emb = np.zeros_like(q_emb)
        for i in range(len(queries)):
            top_idx = ix1[i][0]
            if top_idx == -1:
                expanded_emb[i] = q_emb[i]
            else:
                chunk_emb = emb[top_idx]
                expanded_emb[i] = alpha * q_emb[i] + (1 - alpha) * chunk_emb
        # Normalize
        norms = np.linalg.norm(expanded_emb, axis=1, keepdims=True)
        expanded_emb = expanded_emb / (norms + 1e-8)
        sc2, ix2 = idx.search(expanded_emb.astype('float32'), 100)
        return sc2, ix2

    elif method == 'concat':
        # Re-encode query + top chunk text
        expanded_queries = []
        for i in range(len(queries)):
            top_idx = ix1[i][0]
            if top_idx != -1:
                top_text = chunks[top_idx]['text'][:200]  # limit length
                expanded_queries.append(queries[i] + ' ' + top_text)
            else:
                expanded_queries.append(queries[i])
        eq_emb = model.encode(expanded_queries, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True).astype('float32')
        sc2, ix2 = idx.search(eq_emb, 100)
        return sc2, ix2

    elif method == 'top3_concat':
        expanded_queries = []
        for i in range(len(queries)):
            texts = []
            for j in range(min(3, len(ix1[i]))):
                if ix1[i][j] != -1:
                    texts.append(chunks[ix1[i][j]]['text'][:100])
            expanded_queries.append(queries[i] + ' ' + ' '.join(texts))
        eq_emb = model.encode(expanded_queries, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True).astype('float32')
        sc2, ix2 = idx.search(eq_emb, 100)
        return sc2, ix2

EXPERIMENTS = [
    ('interpolate_a30', 'interpolate', 0.3),
    ('interpolate_a50', 'interpolate', 0.5),
    ('interpolate_a70', 'interpolate', 0.7),
    ('interpolate_a80', 'interpolate', 0.8),
    ('interpolate_a90', 'interpolate', 0.9),
    ('concat_top1', 'concat', 0),
    ('top3_concat', 'top3_concat', 0),
]

best_score = m['FinalScore']
best_name = 'baseline'
best_results = None

for name, method, alpha in EXPERIMENTS:
    print(f'\n=== {name} ===')
    t0 = time.time()
    sc2, ix2 = expand_and_search(queries, qids, idx, chunks, model, method, alpha)

    results = []
    for i in range(len(qids)):
        ca = [{'video_hash': chunks[j]['video_hash'], 'start': chunks[j]['start'],
               'end': chunks[j]['end'], 'score': float(s)}
              for s, j in zip(sc2[i], ix2[i]) if j != -1]
        hits = merge(ca, 10, 0.95)[:5]
        results.append({'query_id': qids[i], 'hits': hits})

    me = evaluate(gt, results)
    elapsed = time.time() - t0
    print(f"  AvgSR={me['AvgSR']:.4f} AvgVR={me['AvgVR']:.4f} FS={me['FinalScore']:.4f} ({elapsed:.0f}s)")

    if me['FinalScore'] > best_score:
        best_score = me['FinalScore']
        best_name = name
        best_results = (method, alpha)
        print(f"  *** NEW BEST ***")

print(f'\nBEST: {best_name} => FinalScore={best_score:.4f}')

# Generate test submission with best
if best_results:
    method, alpha = best_results
    print(f'\nGenerating test submission with {best_name}...')
    test_queries = [str(row['question']).lower().strip() for _, row in test.iterrows()]
    test_ids = [row['query_id'] for _, row in test.iterrows()]
    sc2, ix2 = expand_and_search(test_queries, test_ids, idx, chunks, model, method, alpha)

    fb = list(h2f.values())[0]; rows = []
    for i in range(len(test_ids)):
        ca = [{'video_hash': chunks[j]['video_hash'], 'start': chunks[j]['start'],
               'end': chunks[j]['end'], 'score': float(s)}
              for s, j in zip(sc2[i], ix2[i]) if j != -1]
        hits = merge(ca, 10, 0.95)[:5]
        d = {'query_id': test_ids[i]}
        for rk in range(1, 6):
            if rk <= len(hits):
                h = hits[rk-1]; d[f'video_file_{rk}'] = h2f.get(h['video_hash'], fb)
                d[f'start_{rk}'] = round(h['start'], 1); d[f'end_{rk}'] = round(h['end'], 1)
            else:
                d[f'video_file_{rk}'] = fb; d[f'start_{rk}'] = 0.0; d[f'end_{rk}'] = 1.0
        rows.append(d)
    cols = ['query_id']
    for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
    pd.DataFrame(rows, columns=cols).to_csv('/root/output/submission_query_expansion.csv', index=False)
    print('Saved submission_query_expansion.csv')
else:
    print('Baseline was best, no expansion helped.')

print('Done!')
