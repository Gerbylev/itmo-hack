#!/usr/bin/env python3
"""Full pipeline: download data, finetune E5-large on new transcripts, compare models."""
import pickle, re, os, json, numpy as np, pandas as pd, faiss, torch, random, gc, time
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)
random.seed(42); np.random.seed(42)

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

def build_all_chunks(transcript_by_hash, config=[(30,15),(60,30),(90,45)]):
    chunks = []
    for vh, segs in transcript_by_hash.items():
        for w, s in config:
            for ch in make_chunks(segs, w, s):
                ch['video_hash'] = vh
                chunks.append(ch)
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

def run_model(model, chunks, tq, test, gt, h2f, name, top_k=100, gap=10, shrink=0.7):
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")

    texts = [ch['text'] for ch in chunks]
    print(f"Encoding {len(texts)} chunks...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    idx = faiss.IndexFlatIP(emb.shape[1]); idx.add(emb)

    # Train eval
    print("Evaluating on train...")
    queries = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
    qids = [row['question_id'] for _, row in tq.iterrows()]
    q_emb = model.encode(queries, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True).astype('float32')
    scores, indices = idx.search(q_emb, top_k)

    results = []
    for i in range(len(qids)):
        cands = [{'video_hash': chunks[j]['video_hash'], 'start': chunks[j]['start'],
                  'end': chunks[j]['end'], 'score': float(s)}
                 for s, j in zip(scores[i], indices[i]) if j != -1]
        hits = merge(cands, gap, shrink)[:5]
        results.append({'query_id': qids[i], 'hits': hits})

    m = evaluate(gt, results)
    print(f"\n--- {name} Train Metrics ---")
    for k in (1,3,5):
        print(f"  SR@{k}={m[f'SR@{k}']:.4f}  VR@{k}={m[f'VR@{k}']:.4f}")
    print(f"  AvgSR={m['AvgSR']:.4f}  AvgVR={m['AvgVR']:.4f}  FinalScore={m['FinalScore']:.4f}")

    # Test submission
    print("Generating test submission...")
    tq_test = [str(row['question']).lower().strip() for _, row in test.iterrows()]
    tq_ids = [row['query_id'] for _, row in test.iterrows()]
    qt_emb = model.encode(tq_test, batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True).astype('float32')
    ts, ti = idx.search(qt_emb, top_k)

    rows = []
    fallback = list(h2f.values())[0] if h2f else 'video_unknown'
    for i in range(len(tq_ids)):
        cands = [{'video_hash': chunks[j]['video_hash'], 'start': chunks[j]['start'],
                  'end': chunks[j]['end'], 'score': float(s)}
                 for s, j in zip(ts[i], ti[i]) if j != -1]
        hits = merge(cands, gap, shrink)[:5]
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
    path = f'{WORK}/submission_{name}.csv'
    sub.to_csv(path, index=False)
    print(f"Saved: {path}")

    del emb, idx
    gc.collect(); torch.cuda.empty_cache()
    return m

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
print("Loading data...")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)

# Inspect format
sample_key = list(transcripts.keys())[0]
sample_val = transcripts[sample_key]
print(f"Transcript keys: {len(transcripts)}")
print(f"Sample key: {sample_key}")
print(f"Sample value type: {type(sample_val)}")
if isinstance(sample_val, list) and len(sample_val) > 0:
    print(f"Sample segment: {sample_val[0]}")

transcript_by_hash = {}
for key, segs in transcripts.items():
    vh = extract_hash(key)
    if vh and vh not in SKIP_HASHES:
        transcript_by_hash[vh] = segs

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

chunks = build_all_chunks(transcript_by_hash)
print(f"Total chunks: {len(chunks)}")

# ═══════════════════════════════════════════════════════════
# MODEL 1: E5-large base
# ═══════════════════════════════════════════════════════════
print("\nLoading E5-large base...")
model_e5 = SentenceTransformer('intfloat/multilingual-e5-large', device=DEVICE)
m2 = run_model(model_e5, chunks, tq, test, gt, h2f, 'e5_large_base_new_transcripts', shrink=0.7)
del model_e5; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# FINE-TUNE E5-large on new transcripts
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINE-TUNING E5-large on new transcripts")
print("="*60)

def get_segment_text(segments, start, end, expand=15):
    ws = [s for s in segments if s['end'] > max(0, start - expand) and s['start'] < end + expand]
    if not ws: return ""
    return ' '.join(s['text'].lower().strip() for s in ws)

# Build training pairs
question_positives = defaultdict(list)
for _, row in train.iterrows():
    qid = row['question_id']
    vh = extract_hash(row['video_file'])
    if vh not in transcript_by_hash: continue
    text = get_segment_text(transcript_by_hash[vh], row['start'], row['end'])
    if len(text.split()) < 5: continue
    question_positives[qid].append(text)

chunks_by_video = defaultdict(list)
for ch in chunks:
    chunks_by_video[ch['video_hash']].append(ch)

anchors, positives, negatives = [], [], []
for _, row in tq.iterrows():
    qid, q_en = row['question_id'], row['question_en']
    if qid not in question_positives: continue
    gt_items = gt[qid]
    gt_hashes = {g['video_hash'] for g in gt_items}
    for pos_text in question_positives[qid][:3]:
        neg_text = None
        for g in gt_items:
            vh = g['video_hash']
            if vh not in chunks_by_video: continue
            neg_cands = [c for c in chunks_by_video[vh] if iou(c['start'], c['end'], g['start'], g['end']) < 0.1]
            if neg_cands:
                neg_text = random.choice(neg_cands)['text']; break
        if neg_text is None:
            neg_hashes = [h for h in chunks_by_video if h not in gt_hashes]
            if neg_hashes:
                neg_text = random.choice(chunks_by_video[random.choice(neg_hashes)])['text']
        if neg_text:
            anchors.append(q_en); positives.append(pos_text); negatives.append(neg_text)

# Russian questions too
if 'question_ru' in train.columns:
    train_ru = train[['question_id', 'question_ru']].drop_duplicates('question_id')
    for _, row in train_ru.iterrows():
        qid, q_ru = row['question_id'], row.get('question_ru', '')
        if pd.isna(q_ru) or not q_ru or qid not in question_positives: continue
        gt_items = gt[qid]; gt_hashes = {g['video_hash'] for g in gt_items}
        for pos_text in question_positives[qid][:2]:
            neg_hashes = [h for h in chunks_by_video if h not in gt_hashes]
            if neg_hashes:
                neg_text = random.choice(chunks_by_video[random.choice(neg_hashes)])['text']
                anchors.append(q_ru); positives.append(pos_text); negatives.append(neg_text)

print(f"Training triplets: {len(anchors)}")
train_dataset = Dataset.from_dict({'anchor': anchors, 'positive': positives, 'negative': negatives})

model_e5_ft = SentenceTransformer('intfloat/multilingual-e5-large', device=DEVICE)
model_e5_ft.max_seq_length = 256

loss = losses.MultipleNegativesRankingLoss(model_e5_ft)
effective_batch = 8 * 4
num_steps = (len(train_dataset) // effective_batch) * 4
warmup = int(num_steps * 0.1)
print(f"Steps: {num_steps}, Warmup: {warmup}")

args = SentenceTransformerTrainingArguments(
    output_dir=f'{WORK}/ft_e5_new',
    num_train_epochs=4, per_device_train_batch_size=8,
    gradient_accumulation_steps=4, learning_rate=2e-5,
    warmup_steps=warmup, fp16=True, logging_steps=50,
    save_strategy='epoch', save_total_limit=1,
    dataloader_drop_last=True, report_to='none',
)
trainer = SentenceTransformerTrainer(model=model_e5_ft, args=args,
                                     train_dataset=train_dataset, loss=loss)
trainer.train()
model_e5_ft.save(f'{WORK}/ft_e5_new_best')
print("Fine-tuning done!")

# Upload to HF
print("Uploading to HuggingFace...")
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("olegGerbylev/e5-large-video-retrieval-ft-v2", exist_ok=True)
api.upload_folder(folder_path=f'{WORK}/ft_e5_new_best',
                  repo_id="olegGerbylev/e5-large-video-retrieval-ft-v2",
                  ignore_patterns=["checkpoint-*"])
print("Uploaded to HF!")

# Reload and evaluate
model_e5_ft = SentenceTransformer(f'{WORK}/ft_e5_new_best', device=DEVICE, trust_remote_code=True)
m3 = run_model(model_e5_ft, chunks, tq, test, gt, h2f, 'e5_large_ft_new_transcripts', shrink=0.7)

# Also try different shrink values
for shrink in [0.5, 0.8, 0.95]:
    m_s = run_model(model_e5_ft, chunks, tq, test, gt, h2f,
                    f'e5_large_ft_new_s{int(shrink*100)}', shrink=shrink)

del model_e5_ft; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print("All submissions saved in /root/output/")
print("Done!")
