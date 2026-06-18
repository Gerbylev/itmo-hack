#!/usr/bin/env python3
"""Inference-only: E5-large stage1 + QA span extraction stage2."""
import pickle, re, os, json, numpy as np, pandas as pd, faiss, torch, gc
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda'
SKIP_HASHES = {'7d49c038'}
MAX_LEN = 512

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

def build_transcript_text_with_timestamps(segments):
    text_parts, char_to_time, offset = [], [], 0
    for seg in segments:
        t = seg['text'].strip()
        if not t: continue
        text_parts.append(t)
        char_to_time.append((offset, offset + len(t), seg['start'], seg['end']))
        offset += len(t) + 1
    return ' '.join(text_parts), char_to_time

def time_from_char_offset(c2t, char_pos):
    for cs, ce, ts, te in c2t:
        if cs <= char_pos < ce:
            frac = (char_pos - cs) / max(ce - cs, 1)
            return ts + frac * (te - ts)
    if c2t: return c2t[-1][3]
    return 0

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

# Load data
print("Loading data...")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)
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

# Stage 1: E5-large for video retrieval
print("Loading E5-large...")
e5 = SentenceTransformer('intfloat/multilingual-e5-large', device=DEVICE)

coarse_chunks = []
for vh, segs in transcript_by_hash.items():
    for w, s in [(30.0, 15.0), (60.0, 30.0), (90.0, 45.0)]:
        for ch in make_chunks(segs, w, s):
            ch['video_hash'] = vh
            coarse_chunks.append(ch)
print(f"Coarse chunks: {len(coarse_chunks)}")

coarse_emb = e5.encode([c['text'] for c in coarse_chunks], batch_size=64,
                        show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True).astype('float32')
coarse_idx = faiss.IndexFlatIP(coarse_emb.shape[1])
coarse_idx.add(coarse_emb)

# Pre-encode all queries
print("Encoding train queries...")
queries = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
qids = [row['question_id'] for _, row in tq.iterrows()]
q_emb = e5.encode(queries, batch_size=64, normalize_embeddings=True,
                  convert_to_numpy=True).astype('float32')
all_scores, all_indices = coarse_idx.search(q_emb, 100)

print("Encoding test queries...")
test_queries = [str(row['question']).lower().strip() for _, row in test.iterrows()]
test_qids = [row['query_id'] for _, row in test.iterrows()]
test_q_emb = e5.encode(test_queries, batch_size=64, normalize_embeddings=True,
                       convert_to_numpy=True).astype('float32')
test_scores, test_indices = coarse_idx.search(test_q_emb, 100)

del e5; gc.collect(); torch.cuda.empty_cache()

# Stage 2: QA model
print("Loading QA model...")
qa_model = AutoModelForQuestionAnswering.from_pretrained(f'{WORK}/qa_model_best').to(DEVICE)
qa_tokenizer = AutoTokenizer.from_pretrained(f'{WORK}/qa_model_best')

def predict_span(question, context, c2t):
    inputs = qa_tokenizer(question, context, truncation='only_second',
                          max_length=MAX_LEN, padding='max_length',
                          return_offsets_mapping=True, return_tensors='pt')
    offsets = inputs.pop('offset_mapping')[0]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = qa_model(**inputs)

    input_ids = inputs['input_ids'][0]
    sep_pos = (input_ids == qa_tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
    if len(sep_pos) >= 2:
        ctx_start, ctx_end = int(sep_pos[0]) + 1, int(sep_pos[1]) - 1
    else:
        ctx_start, ctx_end = 1, len(input_ids) - 2

    sl = out.start_logits[0][ctx_start:ctx_end+1]
    el = out.end_logits[0][ctx_start:ctx_end+1]

    si = int(torch.argmax(sl)) + ctx_start
    ei = int(torch.argmax(el)) + ctx_start
    if ei < si: ei = si

    score = float(sl[si - ctx_start] + el[ei - ctx_start])
    sc, ec = int(offsets[si][0]), int(offsets[ei][1])

    start_t = time_from_char_offset(c2t, sc)
    end_t = time_from_char_offset(c2t, ec)
    if end_t <= start_t: end_t = start_t + 30
    return start_t, end_t, score

def run_two_stage(query_idx, scores, indices, query_text, top_videos=10):
    video_scores = {}
    for s, j in zip(scores[query_idx][:100], indices[query_idx][:100]):
        if j == -1: continue
        vh = coarse_chunks[j]['video_hash']
        if vh not in video_scores or s > video_scores[vh]:
            video_scores[vh] = float(s)
    top_vhs = sorted(video_scores, key=video_scores.get, reverse=True)[:top_videos]

    hits = []
    for vh in top_vhs:
        if vh not in transcript_by_hash: continue
        segs = transcript_by_hash[vh]
        full_text, c2t = build_transcript_text_with_timestamps(segs)
        if len(full_text) < 10: continue

        try:
            st, et, sc = predict_span(query_text, full_text, c2t)
            hits.append({'video_hash': vh, 'start': round(st,1), 'end': round(et,1),
                        'score': sc * video_scores[vh]})
        except:
            hits.append({'video_hash': vh, 'start': 0, 'end': 60,
                        'score': video_scores[vh] * 0.1})

    hits.sort(key=lambda x: x['score'], reverse=True)

    # Fill to 5 with coarse fallback
    seen = {h['video_hash'] for h in hits}
    for s, j in zip(scores[query_idx], indices[query_idx]):
        if len(hits) >= 5: break
        if j == -1: continue
        c = coarse_chunks[j]
        if c['video_hash'] not in seen:
            hits.append({'video_hash': c['video_hash'], 'start': c['start'],
                        'end': c['end'], 'score': float(s) * 0.01})
            seen.add(c['video_hash'])
    return hits[:5]

# Evaluate on train
print("\nEvaluating on train...")
results = []
for i in tqdm(range(len(qids)), desc='Train eval'):
    hits = run_two_stage(i, all_scores, all_indices, queries[i])
    results.append({'query_id': qids[i], 'hits': hits})

m = evaluate(gt, results)
print(f"\n=== Two-Stage QA Results ===")
for k in (1, 3, 5):
    print(f"  SR@{k}={m[f'SR@{k}']:.4f}  VR@{k}={m[f'VR@{k}']:.4f}")
print(f"  AvgSR={m['AvgSR']:.4f}  AvgVR={m['AvgVR']:.4f}  FinalScore={m['FinalScore']:.4f}")

# Generate test submission
print("\nGenerating test submission...")
test_results = []
for i in tqdm(range(len(test_qids)), desc='Test'):
    hits = run_two_stage(i, test_scores, test_indices, test_queries[i])
    test_results.append({'query_id': test_qids[i], 'hits': hits})

rows = []
for r in test_results:
    d = {'query_id': r['query_id']}
    for rk in range(1, 6):
        if rk <= len(r['hits']):
            h = r['hits'][rk-1]
            d[f'video_file_{rk}'] = h2f.get(h['video_hash'], h['video_hash'])
            d[f'start_{rk}'] = round(h['start'], 1)
            d[f'end_{rk}'] = round(h['end'], 1)
        else:
            fallback = list(h2f.values())[0]
            d[f'video_file_{rk}'] = fallback
            d[f'start_{rk}'] = 0.0
            d[f'end_{rk}'] = 1.0
    rows.append(d)
cols = ['query_id']
for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
sub = pd.DataFrame(rows, columns=cols)
sub.to_csv(f'{WORK}/submission_qa_span.csv', index=False)
print(f"Saved: {WORK}/submission_qa_span.csv")
print("Done!")
