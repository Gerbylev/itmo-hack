#!/usr/bin/env python3
"""Grid search over shrink, gap, top_k for E5-large fine-tuned."""
import pickle, re, numpy as np, pandas as pd, faiss, torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = "/root/data/video-rag"
WORK = "/root/output"
SKIP_HASHES = {"7d49c038"}

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def make_chunks(segments, window, step):
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
                    bc = (best['start']+best['end'])/2
                    bh = (best['end']-best['start'])/2
                    ch = (cur['end']-cur['start'])/2
                    nh = ch*(1-shrink)+bh*shrink
                    cur['start'] = max(cur['start'], bc-nh)
                    cur['end'] = min(cur['end'], bc+nh)
                merged.append(cur)
                cur, best = nxt.copy(), nxt.copy()
        if shrink > 0:
            bc = (best['start']+best['end'])/2
            bh = (best['end']-best['start'])/2
            ch = (cur['end']-cur['start'])/2
            nh = ch*(1-shrink)+bh*shrink
            cur['start'] = max(cur['start'], bc-nh)
            cur['end'] = min(cur['end'], bc+nh)
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
with open(f'{BASE}/transcripts.pkl','rb') as f: transcripts=pickle.load(f)
train=pd.read_csv(f'{BASE}/train/train_qa.csv')
gt={}
for _,row in train.iterrows():
    qid,vh=row['question_id'],extract_hash(row['video_file'])
    gt.setdefault(qid,[]).append({'video_hash':vh,'start':row['start'],'end':row['end']})
tq=train[['question_id','question_en']].drop_duplicates('question_id')

th={}
for key,segs in transcripts.items():
    vh=extract_hash(key)
    if vh and vh not in SKIP_HASHES: th[vh]=segs

model=SentenceTransformer(f'{WORK}/finetuned_e5_large',device='cuda',trust_remote_code=True)

chunks=[]
for vh,segs in th.items():
    for w,s in [(30.0,15.0),(60.0,30.0)]:
        for ch in make_chunks(segs,w,s):
            ch['video_hash']=vh; chunks.append(ch)

print(f'Chunks: {len(chunks)}')
emb=model.encode([c['text'] for c in chunks],batch_size=64,show_progress_bar=True,
                 normalize_embeddings=True,convert_to_numpy=True).astype('float32')
idx=faiss.IndexFlatIP(emb.shape[1]); idx.add(emb)

print('Encoding queries...')
queries=[row['question_en'].lower().strip() for _,row in tq.iterrows()]
qids=[row['question_id'] for _,row in tq.iterrows()]
q_emb=model.encode(queries,batch_size=64,normalize_embeddings=True,
                   convert_to_numpy=True).astype('float32')

print('Searching...')
all_scores,all_indices=idx.search(q_emb,200)

print('\nGrid search...')
best_fs,best_cfg=0,None
for shrink in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for gap in [0,5,10,15,20]:
        for top_k in [50,100,200]:
            results=[]
            for i in range(len(qids)):
                cands=[]
                for s,j in zip(all_scores[i][:top_k],all_indices[i][:top_k]):
                    if j==-1:continue
                    c=chunks[j]
                    cands.append({'video_hash':c['video_hash'],'start':c['start'],'end':c['end'],'score':float(s)})
                hits=merge(cands,gap,shrink)[:5]
                results.append({'query_id':qids[i],'hits':hits})
            m=evaluate(gt,results)
            fs=m['FinalScore']
            if fs>best_fs:
                best_fs=fs
                best_cfg={'shrink':shrink,'gap':gap,'top_k':top_k}
                avgsr=m['AvgSR']
                avgvr=m['AvgVR']
                print(f'  NEW BEST: shrink={shrink} gap={gap} top_k={top_k} => AvgSR={avgsr:.4f} AvgVR={avgvr:.4f} FS={fs:.4f}')

print(f'\nBEST: {best_cfg} => FinalScore={best_fs:.4f}')

# Generate submission with best config
print('\nGenerating submission...')
test=pd.read_csv(f'{BASE}/test/test.csv')
vf=pd.read_csv(f'{BASE}/video_files.csv')
h2f={}
for p in vf['video_path']:
    h=extract_hash(p); fn=re.sub(r'\.\w+$','',p.split('/')[-1])
    if h: h2f[h]=fn

tq_test=[str(row['question']).lower().strip() for _,row in test.iterrows()]
tq_ids=[row['query_id'] for _,row in test.iterrows()]
qt_emb=model.encode(tq_test,batch_size=64,normalize_embeddings=True,
                    convert_to_numpy=True).astype('float32')
ts,ti=idx.search(qt_emb,best_cfg['top_k'])

rows=[]
for i in range(len(tq_ids)):
    cands=[]
    for s,j in zip(ts[i],ti[i]):
        if j==-1:continue
        c=chunks[j]
        cands.append({'video_hash':c['video_hash'],'start':c['start'],'end':c['end'],'score':float(s)})
    hits=merge(cands,best_cfg['gap'],best_cfg['shrink'])[:5]
    d={'query_id':tq_ids[i]}
    for rk in range(1,6):
        if rk<=len(hits):
            h=hits[rk-1]
            d[f'video_file_{rk}']=h2f.get(h['video_hash'],h['video_hash'])
            d[f'start_{rk}']=round(h['start'],1)
            d[f'end_{rk}']=round(h['end'],1)
        else:
            d[f'video_file_{rk}'],d[f'start_{rk}'],d[f'end_{rk}']='',0.0,0.0
    rows.append(d)
cols=['query_id']
for rk in range(1,6): cols+=[f'video_file_{rk}',f'start_{rk}',f'end_{rk}']
sub=pd.DataFrame(rows,columns=cols)
sub.to_csv(f'{WORK}/submission_gridsearch_best.csv',index=False)
print(f'Saved: {WORK}/submission_gridsearch_best.csv')
print('Done!')
