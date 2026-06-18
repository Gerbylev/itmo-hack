#!/usr/bin/env python3
"""Fine-tune embedding models on video-rag training data and evaluate."""

import pickle, re, os, json, numpy as np, pandas as pd, faiss, torch, time, random, gc
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────
BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)
random.seed(42)
np.random.seed(42)

# ── Helpers ─────────────────────────────────────────────────────
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

def generate_submission(model, chunks, index, test_df, h2f, work, name, top_k=100, gap=10, shrink=0.7):
    test_r = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f'Submission {name}'):
        q = str(row['question']).lower().strip()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qv, index, chunks, top_k, 5, gap, shrink)
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
    path = f'{work}/submission_{name}.csv'
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    print(f"  Submission saved: {path}")
    return path

# ── Load data ───────────────────────────────────────────────────
print(f"Device: {DEVICE}")
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

# ── Build transcript lookup by video hash ───────────────────────
print("Building transcript index...")
transcript_by_hash = {}
for key, segments in transcripts.items():
    vh = extract_hash(key)
    if vh and vh not in SKIP_HASHES:
        transcript_by_hash[vh] = segments

# ── STEP 1: Create training pairs ──────────────────────────────
print("\n=== Creating training pairs ===")

def get_segment_text(segments, start, end):
    ws = [s for s in segments if s['end'] > start and s['start'] < end]
    if not ws: return ""
    return ' '.join(clean_text(s['text']) for s in ws)

def get_segment_text_expanded(segments, start, end, expand=15):
    return get_segment_text(segments, max(0, start - expand), end + expand)

# Build positive pairs
question_positives = defaultdict(list)
for _, row in train.iterrows():
    qid = row['question_id']
    vh = extract_hash(row['video_file'])
    if vh not in transcript_by_hash: continue
    segments = transcript_by_hash[vh]
    text = get_segment_text_expanded(segments, row['start'], row['end'])
    if len(text.split()) < 5: continue
    question_positives[qid].append(text)

has_ru = 'question_ru' in train.columns
print(f"Has Russian questions: {has_ru}")

# Build all chunk texts for hard negative mining
print("Building chunks for negative mining...")
all_neg_chunks = build_chunks([(30.0, 15.0), (60.0, 30.0)], transcripts)
print(f"Total chunks for neg mining: {len(all_neg_chunks)}")

chunks_by_video = defaultdict(list)
for ch in all_neg_chunks:
    chunks_by_video[ch['video_hash']].append(ch)

# Create dataset for sentence-transformers v3+ using Dataset format
print("Creating training dataset...")
anchors = []
positives = []
negatives = []

for _, row in tq.iterrows():
    qid = row['question_id']
    q_en = row['question_en']
    if qid not in question_positives: continue

    pos_texts = question_positives[qid]
    gt_items = gt[qid]
    gt_hashes = {g['video_hash'] for g in gt_items}

    for pos_text in pos_texts[:3]:
        # Hard negative: same video, wrong time
        neg_text = None
        for g in gt_items:
            vh = g['video_hash']
            if vh not in chunks_by_video: continue
            neg_cands = [c for c in chunks_by_video[vh]
                        if iou(c['start'], c['end'], g['start'], g['end']) < 0.1]
            if neg_cands:
                neg_text = random.choice(neg_cands)['text']
                break

        if neg_text is None:
            # Cross-video negative
            all_hashes = list(chunks_by_video.keys())
            neg_hashes = [h for h in all_hashes if h not in gt_hashes]
            if neg_hashes:
                neg_vh = random.choice(neg_hashes)
                neg_text = random.choice(chunks_by_video[neg_vh])['text']

        if neg_text:
            anchors.append(q_en)
            positives.append(pos_text)
            negatives.append(neg_text)

# Also add Russian questions
if has_ru:
    train_ru = train[['question_id', 'question_ru']].drop_duplicates('question_id')
    for _, row in train_ru.iterrows():
        qid = row['question_id']
        q_ru = row.get('question_ru', '')
        if pd.isna(q_ru) or not q_ru or qid not in question_positives: continue

        pos_texts = question_positives[qid]
        gt_items = gt[qid]
        gt_hashes = {g['video_hash'] for g in gt_items}

        for pos_text in pos_texts[:2]:
            neg_text = None
            all_hashes = list(chunks_by_video.keys())
            neg_hashes = [h for h in all_hashes if h not in gt_hashes]
            if neg_hashes:
                neg_vh = random.choice(neg_hashes)
                neg_text = random.choice(chunks_by_video[neg_vh])['text']
            if neg_text:
                anchors.append(q_ru)
                positives.append(pos_text)
                negatives.append(neg_text)

print(f"Total training triplets: {len(anchors)}")

train_dataset = Dataset.from_dict({
    'anchor': anchors,
    'positive': positives,
    'negative': negatives,
})

# Also create a pairs-only dataset (for MNR loss which uses in-batch negatives)
pair_anchors = []
pair_positives = []

for _, row in tq.iterrows():
    qid = row['question_id']
    q_en = row['question_en']
    if qid not in question_positives: continue
    for pos_text in question_positives[qid][:3]:
        pair_anchors.append(q_en)
        pair_positives.append(pos_text)

if has_ru:
    for _, row in train_ru.iterrows():
        qid = row['question_id']
        q_ru = row.get('question_ru', '')
        if pd.isna(q_ru) or not q_ru or qid not in question_positives: continue
        for pos_text in question_positives[qid][:2]:
            pair_anchors.append(q_ru)
            pair_positives.append(pos_text)

pair_dataset = Dataset.from_dict({
    'anchor': pair_anchors,
    'positive': pair_positives,
})
print(f"Total pair examples: {len(pair_anchors)}")

# ── Eval and submission helpers ─────────────────────────────────
BEST_CONFIG = {'top_k': 100, 'gap': 10, 'shrink': 0.7}
CHUNK_CONFIG = [(30.0, 15.0), (60.0, 30.0)]

def eval_model(model, model_name_short):
    print(f"\n  Evaluating {model_name_short}...")
    chunks = build_chunks(CHUNK_CONFIG, transcripts)
    texts = [ch['text'] for ch in chunks]
    print(f"  Encoding {len(texts)} chunks...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    tr = []
    for _, row in tqdm(tq.iterrows(), total=len(tq), desc='Train eval'):
        q = row['question_en'].lower().strip()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qv, idx, chunks, **BEST_CONFIG, top_n=5)
        tr.append({'query_id': row['question_id'], 'hits': hits})

    m = evaluate(gt, tr)
    print(f"\n  === {model_name_short} Train Metrics ===")
    for k in (1, 3, 5):
        print(f"    SR@{k}={m[f'SR@{k}']:.4f}  VR@{k}={m[f'VR@{k}']:.4f}")
    print(f"    AvgSR={m['AvgSR']:.4f}  AvgVR={m['AvgVR']:.4f}  FinalScore={m['FinalScore']:.4f}")

    return chunks, idx, m

def finetune_model(model_name, save_name, epochs=3, lr=2e-5, batch_size=8,
                   use_triplets=True, use_mnrl=True, grad_accum=4):
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_name, device=DEVICE, trust_remote_code=True)
    # Truncate long texts to save memory
    model.max_seq_length = 256

    save_path = f'{WORK}/finetuned_{save_name}'

    if use_mnrl:
        loss = losses.MultipleNegativesRankingLoss(model)
        ds = train_dataset if use_triplets else pair_dataset
    else:
        loss = losses.TripletLoss(model)
        ds = train_dataset

    effective_batch = batch_size * grad_accum
    num_steps = (len(ds) // effective_batch) * epochs
    warmup_steps = int(num_steps * 0.1)

    print(f"  Dataset size: {len(ds)}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}x{grad_accum}={effective_batch}")
    print(f"  Total steps: {num_steps}, Warmup: {warmup_steps}")
    print(f"  Max seq length: {model.max_seq_length}")

    args = SentenceTransformerTrainingArguments(
        output_dir=save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        fp16=True,
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=1,
        dataloader_drop_last=True,
        report_to='none',
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        loss=loss,
    )

    trainer.train()
    model.save(save_path)
    print(f"  Model saved to {save_path}")

    # Reload
    model = SentenceTransformer(save_path, device=DEVICE, trust_remote_code=True)

    # Evaluate
    chunks, idx, metrics = eval_model(model, f'{save_name}_finetuned')

    # Generate submission
    sub_path = generate_submission(model, chunks, idx, test, h2f, WORK,
                                   f'ft_{save_name}', **BEST_CONFIG)

    del idx
    gc.collect()
    torch.cuda.empty_cache()

    return model, metrics, sub_path

# ── Run experiments ─────────────────────────────────────────────
results_summary = []

# ── Baseline BGE-M3 ────────────────────────────────────────────
print("\n\n" + "="*70)
print("MODEL 1: BAAI/bge-m3 (baseline, no fine-tuning)")
print("="*70)

model_bge = SentenceTransformer('BAAI/bge-m3', device=DEVICE, trust_remote_code=True)
_, _, baseline_bge_metrics = eval_model(model_bge, 'bge-m3_baseline')
results_summary.append({'model': 'bge-m3_baseline', **{k: round(v,4) for k,v in baseline_bge_metrics.items()}})
del model_bge; gc.collect(); torch.cuda.empty_cache()

# ── Fine-tune BGE-M3 (MNRL with triplets) ──────────────────────
ft_bge_model, ft_bge_metrics, ft_bge_sub = finetune_model(
    'BAAI/bge-m3', 'bge_m3', epochs=4, lr=2e-5, batch_size=8,
    use_triplets=True, use_mnrl=True
)
results_summary.append({'model': 'bge-m3_ft_mnrl', **{k: round(v,4) for k,v in ft_bge_metrics.items()}})
del ft_bge_model; gc.collect(); torch.cuda.empty_cache()

# ── Baseline E5-large ──────────────────────────────────────────
print("\n\n" + "="*70)
print("MODEL 2: intfloat/multilingual-e5-large (baseline, no fine-tuning)")
print("="*70)

model_e5 = SentenceTransformer('intfloat/multilingual-e5-large', device=DEVICE)
_, _, baseline_e5_metrics = eval_model(model_e5, 'e5-large_baseline')
results_summary.append({'model': 'e5-large_baseline', **{k: round(v,4) for k,v in baseline_e5_metrics.items()}})
del model_e5; gc.collect(); torch.cuda.empty_cache()

# ── Fine-tune E5-large ─────────────────────────────────────────
ft_e5_model, ft_e5_metrics, ft_e5_sub = finetune_model(
    'intfloat/multilingual-e5-large', 'e5_large', epochs=4, lr=2e-5, batch_size=8,
    use_triplets=True, use_mnrl=True
)
results_summary.append({'model': 'e5-large_ft_mnrl', **{k: round(v,4) for k,v in ft_e5_metrics.items()}})
del ft_e5_model; gc.collect(); torch.cuda.empty_cache()

# ── Fine-tune BGE-M3 v2 (more epochs, lower LR) ───────────────
ft_bge2_model, ft_bge2_metrics, ft_bge2_sub = finetune_model(
    'BAAI/bge-m3', 'bge_m3_v2', epochs=5, lr=1e-5, batch_size=8,
    use_triplets=True, use_mnrl=True
)
results_summary.append({'model': 'bge-m3_ft_v2', **{k: round(v,4) for k,v in ft_bge2_metrics.items()}})
del ft_bge2_model; gc.collect(); torch.cuda.empty_cache()

# ── Final Summary ───────────────────────────────────────────────
print("\n\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
df_results = pd.DataFrame(results_summary)
print(df_results.to_string(index=False))

best = df_results.loc[df_results['FinalScore'].idxmax()]
print(f"\nBEST MODEL: {best['model']} with FinalScore={best['FinalScore']:.4f}")

df_results.to_csv(f'{WORK}/finetune_summary.csv', index=False)
print(f"\nSummary saved to {WORK}/finetune_summary.csv")
print("Done!")
