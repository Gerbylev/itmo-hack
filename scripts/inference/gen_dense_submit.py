#!/usr/bin/env python3
"""Generate submission with best dense config: 30/60/90 step 5/10/15 shrink=0.95"""
import pickle, re, numpy as np, pandas as pd, faiss, torch
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

with open(f'{BASE}/transcripts.pkl','rb') as f: tr=pickle.load(f)
th={}
for k,s in tr.items():
    vh=xh(k)
    if vh and vh not in SKIP: th[vh]=s

test=pd.read_csv(f'{BASE}/test/test.csv')
vf=pd.read_csv(f'{BASE}/video_files.csv')
h2f={}
for p in vf['video_path']:
    h=xh(p);fn=re.sub(r'\.\w+$','',p.split('/')[-1])
    if h:h2f[h]=fn

print('Loading model...')
model=SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2',device='cuda',trust_remote_code=True)

chunks=[]
for vh,segs in th.items():
    for w,s in [(30,5),(60,10),(90,15)]:
        for ch in mkc(segs,w,s):
            ch['video_hash']=vh;chunks.append(ch)
print(f'Chunks: {len(chunks)}')

emb=model.encode([c['text'] for c in chunks],batch_size=64,show_progress_bar=True,
                 normalize_embeddings=True,convert_to_numpy=True).astype('float32')
idx=faiss.IndexFlatIP(emb.shape[1]);idx.add(emb)

tq=[str(row['question']).lower().strip() for _,row in test.iterrows()]
ids=[row['query_id'] for _,row in test.iterrows()]
qe=model.encode(tq,batch_size=64,normalize_embeddings=True,convert_to_numpy=True).astype('float32')
sc,ix=idx.search(qe,100)

fb=list(h2f.values())[0];rows=[]
for i in range(len(ids)):
    ca=[{'video_hash':chunks[j]['video_hash'],'start':chunks[j]['start'],'end':chunks[j]['end'],'score':float(s)} for s,j in zip(sc[i],ix[i]) if j!=-1]
    hits=merge(ca,10,0.95)[:5]
    d={'query_id':ids[i]}
    for rk in range(1,6):
        if rk<=len(hits):
            h=hits[rk-1];d[f'video_file_{rk}']=h2f.get(h['video_hash'],fb);d[f'start_{rk}']=round(h['start'],1);d[f'end_{rk}']=round(h['end'],1)
        else:
            d[f'video_file_{rk}']=fb;d[f'start_{rk}']=0.0;d[f'end_{rk}']=1.0
    rows.append(d)
cols=['query_id']
for rk in range(1,6):cols+=[f'video_file_{rk}',f'start_{rk}',f'end_{rk}']
pd.DataFrame(rows,columns=cols).to_csv('/root/output/submission_dense_best.csv',index=False)
print('DONE')
