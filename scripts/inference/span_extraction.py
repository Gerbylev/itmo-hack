#!/usr/bin/env python3
"""
Span Extraction (NER-like) for Video Fragment Retrieval.
Given a question + full video transcript, predict which token span is the answer.
Maps token spans back to timestamps via Whisper segment timecodes.

Approach: Fine-tune a QA model (like mBERT/XLM-R) on extractive QA task where:
- Context = video transcript (concatenated Whisper segments with timestamps)
- Question = the query
- Answer span = tokens from the GT segment (start-end)
"""

import pickle, re, os, json, numpy as np, pandas as pd, torch, random, gc
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, DefaultDataCollator
)
from datasets import Dataset
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)
random.seed(42)
np.random.seed(42)

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

# ── Load data ───────────────────────────────────────────────────
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
    h = extract_hash(p)
    fn = re.sub(r'\.\w+$', '', p.split('/')[-1])
    if h: h2f[h] = fn

# ── Build QA training data ──────────────────────────────────────
print("Building QA training data...")

MODEL_NAME = 'deepset/xlm-roberta-base-squad2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 512
DOC_STRIDE = 128

def build_transcript_text_with_timestamps(segments):
    """Build full transcript text and keep mapping: char_offset -> timestamp"""
    text_parts = []
    char_to_time = []  # list of (char_start, char_end, seg_start_time, seg_end_time)
    offset = 0
    for seg in segments:
        t = seg['text'].strip()
        if not t:
            continue
        text_parts.append(t)
        char_to_time.append((offset, offset + len(t), seg['start'], seg['end']))
        offset += len(t) + 1  # +1 for space
    full_text = ' '.join(text_parts)
    return full_text, char_to_time

def find_answer_span(char_to_time, gt_start, gt_end):
    """Find char offsets in full text that correspond to GT time segment."""
    answer_start = None
    answer_end = None
    for char_s, char_e, t_s, t_e in char_to_time:
        # Segment overlaps with GT
        if t_e > gt_start and t_s < gt_end:
            if answer_start is None:
                answer_start = char_s
            answer_end = char_e
    return answer_start, answer_end

def time_from_char_offset(char_to_time, char_pos):
    """Map a character position back to a timestamp."""
    for char_s, char_e, t_s, t_e in char_to_time:
        if char_s <= char_pos < char_e:
            # Interpolate within segment
            frac = (char_pos - char_s) / max(char_e - char_s, 1)
            return t_s + frac * (t_e - t_s)
    # If beyond last, return last end time
    if char_to_time:
        return char_to_time[-1][3]
    return 0

# Build training examples
qa_examples = []
skipped = 0

# Group by question_id to handle multiple GT segments
gt_by_qid = {}
for _, row in train.iterrows():
    qid = row['question_id']
    vh = extract_hash(row['video_file'])
    gt_by_qid.setdefault(qid, []).append({
        'video_hash': vh, 'start': row['start'], 'end': row['end'],
        'question_en': row['question_en'],
        'question_ru': row.get('question_ru', '')
    })

for qid, gt_items in tqdm(gt_by_qid.items(), desc='Building QA data'):
    for gt in gt_items:
        vh = gt['video_hash']
        if vh not in transcript_by_hash:
            skipped += 1
            continue

        segments = transcript_by_hash[vh]
        full_text, char_to_time = build_transcript_text_with_timestamps(segments)

        if len(full_text) < 10:
            skipped += 1
            continue

        ans_start, ans_end = find_answer_span(char_to_time, gt['start'], gt['end'])
        if ans_start is None:
            skipped += 1
            continue

        answer_text = full_text[ans_start:ans_end]
        if len(answer_text.strip()) < 5:
            skipped += 1
            continue

        # If transcript is too long, take a window around the answer
        if len(full_text) > 3000:
            # Center window around answer
            window_size = 2500
            center = (ans_start + ans_end) // 2
            ctx_start = max(0, center - window_size // 2)
            ctx_end = min(len(full_text), center + window_size // 2)

            # Adjust answer offsets
            new_ans_start = ans_start - ctx_start
            new_ans_end = ans_end - ctx_start
            context = full_text[ctx_start:ctx_end]

            # Adjust char_to_time
            new_c2t = []
            for cs, ce, ts, te in char_to_time:
                if ce > ctx_start and cs < ctx_end:
                    new_c2t.append((max(cs - ctx_start, 0), min(ce - ctx_start, ctx_end - ctx_start), ts, te))
        else:
            context = full_text
            new_ans_start = ans_start
            new_ans_end = ans_end
            new_c2t = char_to_time

        # English question
        qa_examples.append({
            'question': gt['question_en'],
            'context': context,
            'answer_start': new_ans_start,
            'answer_text': context[new_ans_start:new_ans_end],
            'video_hash': vh,
            'gt_start': gt['start'],
            'gt_end': gt['end'],
            'char_to_time': json.dumps([(cs, ce, ts, te) for cs, ce, ts, te in new_c2t]),
        })

        # Russian question (if available)
        q_ru = gt.get('question_ru', '')
        if q_ru and not pd.isna(q_ru):
            qa_examples.append({
                'question': q_ru,
                'context': context,
                'answer_start': new_ans_start,
                'answer_text': context[new_ans_start:new_ans_end],
                'video_hash': vh,
                'gt_start': gt['start'],
                'gt_end': gt['end'],
                'char_to_time': json.dumps([(cs, ce, ts, te) for cs, ce, ts, te in new_c2t]),
            })

print(f"QA examples: {len(qa_examples)}, skipped: {skipped}")

# ── Tokenize for QA ────────────────────────────────────────────
print("Tokenizing...")

def prepare_features(examples):
    """Tokenize one example at a time (no overflow mapping needed)."""
    all_input_ids = []
    all_attention_mask = []
    all_start_positions = []
    all_end_positions = []

    for idx in range(len(examples['question'])):
        q = examples['question'][idx]
        ctx = examples['context'][idx]
        ans_start_char = examples['answer_start'][idx]
        ans_text = examples['answer_text'][idx]
        ans_end_char = ans_start_char + len(ans_text)

        tokenized = tokenizer(
            q, ctx,
            truncation='only_second',
            max_length=MAX_LEN,
            padding='max_length',
            return_offsets_mapping=True,
        )

        offsets = tokenized['offset_mapping']
        input_ids = tokenized['input_ids']

        # Find context token boundaries
        # Tokens: [CLS] question [SEP] context [SEP] [PAD]...
        sep_id = tokenizer.sep_token_id
        sep_positions = [i for i, tid in enumerate(input_ids) if tid == sep_id]
        if len(sep_positions) >= 2:
            ctx_start = sep_positions[0] + 1
            ctx_end = sep_positions[1] - 1
        elif len(sep_positions) == 1:
            ctx_start = sep_positions[0] + 1
            ctx_end = len(input_ids) - 2
        else:
            ctx_start = 1
            ctx_end = len(input_ids) - 2

        # Find start/end tokens for answer
        tok_s = 0
        tok_e = 0
        found = False

        for j in range(ctx_start, min(ctx_end + 1, len(offsets))):
            os_j, oe_j = offsets[j]
            if os_j is None:
                continue
            if not found and oe_j > ans_start_char:
                tok_s = j
                found = True
            if found and os_j < ans_end_char:
                tok_e = j

        if not found:
            tok_s = 0
            tok_e = 0

        all_input_ids.append(tokenized['input_ids'])
        all_attention_mask.append(tokenized['attention_mask'])
        all_start_positions.append(tok_s)
        all_end_positions.append(tok_e)

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'start_positions': all_start_positions,
        'end_positions': all_end_positions,
    }

# Split train/val
random.shuffle(qa_examples)
val_size = min(500, len(qa_examples) // 5)
train_examples = qa_examples[val_size:]
val_examples = qa_examples[:val_size]

train_ds = Dataset.from_list(train_examples)
val_ds = Dataset.from_list(val_examples)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_tokenized = train_ds.map(prepare_features, batched=True,
                                remove_columns=train_ds.column_names,
                                batch_size=100)
val_tokenized = val_ds.map(prepare_features, batched=True,
                            remove_columns=val_ds.column_names,
                            batch_size=100)

print(f"Tokenized train: {len(train_tokenized)}, val: {len(val_tokenized)}")

# ── Train QA model ──────────────────────────────────────────────
print(f"\nTraining QA model: {MODEL_NAME}")

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
model.to(DEVICE)

training_args = TrainingArguments(
    output_dir=f'{WORK}/qa_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to='none',
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DefaultDataCollator(),
)

trainer.train()
model.save_pretrained(f'{WORK}/qa_model_best')
tokenizer.save_pretrained(f'{WORK}/qa_model_best')
print("QA model saved")

# ── Two-stage inference ─────────────────────────────────────────
# Stage 1: E5-large-ft finds top videos
# Stage 2: QA model extracts exact span within each video transcript
print("\n=== Two-Stage Inference ===")

# Load E5-large-ft for stage 1
print("Loading E5-large-ft for video retrieval...")
e5_model_path = 'intfloat/multilingual-e5-large'
print("Using base E5-large for stage 1")

del model
gc.collect(); torch.cuda.empty_cache()

e5_model = SentenceTransformer(e5_model_path, device=DEVICE, trust_remote_code=True)

# Build coarse chunks for video retrieval
def make_chunks(segments, window, step):
    chunks, t = [], segments[0]['start']
    while t < segments[-1]['end']:
        ws = [s for s in segments if s['end'] > t and s['start'] < t + window]
        if ws:
            chunks.append({'start': ws[0]['start'], 'end': ws[-1]['end'],
                          'text': ' '.join(s['text'].lower().strip() for s in ws)})
        t += step
    return chunks

coarse_chunks = []
for vh, segs in transcript_by_hash.items():
    for w, s in [(30.0, 15.0), (60.0, 30.0), (90.0, 45.0)]:
        for ch in make_chunks(segs, w, s):
            ch['video_hash'] = vh
            coarse_chunks.append(ch)

print(f"Coarse chunks: {len(coarse_chunks)}")
coarse_emb = e5_model.encode([c['text'] for c in coarse_chunks], batch_size=64,
                              show_progress_bar=True, normalize_embeddings=True,
                              convert_to_numpy=True).astype('float32')
coarse_idx = faiss.IndexFlatIP(coarse_emb.shape[1])
coarse_idx.add(coarse_emb)

del e5_model; gc.collect(); torch.cuda.empty_cache()

# Load QA model for stage 2
print("Loading QA model for span extraction...")
qa_model = AutoModelForQuestionAnswering.from_pretrained(f'{WORK}/qa_model_best').to(DEVICE)
qa_tokenizer = AutoTokenizer.from_pretrained(f'{WORK}/qa_model_best')

def predict_span(question, context, char_to_time):
    """Run QA model and return predicted start/end time."""
    inputs = qa_tokenizer(
        question, context,
        truncation='only_second', max_length=MAX_LEN,
        return_offsets_mapping=True, padding='max_length',
        return_tensors='pt'
    )

    offset_mapping = inputs.pop('offset_mapping')

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = qa_model(**inputs)

    best_score = -float('inf')
    best_start_char = 0
    best_end_char = len(context)

    for i in range(inputs['input_ids'].shape[0]):
        start_logits = outputs.start_logits[i]
        end_logits = outputs.end_logits[i]
        offsets = offset_mapping[i]
        sequence_ids = inputs['input_ids'][i]

        # Mask non-context tokens
        # Find context boundaries
        input_ids = inputs['input_ids'][i]
        sep_positions = (input_ids == qa_tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 2:
            ctx_token_start = int(sep_positions[0]) + 1
            ctx_token_end = int(sep_positions[1])
        else:
            continue

        # Get top start/end positions
        start_probs = torch.softmax(start_logits[ctx_token_start:ctx_token_end+1], dim=0)
        end_probs = torch.softmax(end_logits[ctx_token_start:ctx_token_end+1], dim=0)

        # Top 10 starts and ends
        top_starts = torch.topk(start_probs, min(10, len(start_probs)))
        top_ends = torch.topk(end_probs, min(10, len(end_probs)))

        for si, s_score in zip(top_starts.indices, top_starts.values):
            for ei, e_score in zip(top_ends.indices, top_ends.values):
                si_abs = int(si) + ctx_token_start
                ei_abs = int(ei) + ctx_token_start
                if ei_abs < si_abs or (ei_abs - si_abs) > 150:
                    continue
                score = float(s_score * e_score)
                if score > best_score:
                    best_score = score
                    sc = int(offsets[si_abs][0])
                    ec = int(offsets[ei_abs][1])
                    best_start_char = sc
                    best_end_char = ec

    # Map char positions to timestamps
    c2t = json.loads(char_to_time) if isinstance(char_to_time, str) else char_to_time
    start_time = time_from_char_offset(c2t, best_start_char)
    end_time = time_from_char_offset(c2t, best_end_char)

    if end_time <= start_time:
        end_time = start_time + 30  # fallback

    return start_time, end_time, best_score

def two_stage_predict(question, coarse_idx, coarse_chunks, transcript_by_hash,
                      top_k_videos=10, coarse_top_k=100):
    """Stage 1: find videos, Stage 2: extract spans."""
    # Stage 1: encode query with simple model (reuse coarse embeddings approach)
    # We need to re-encode query... but E5 model is unloaded.
    # Use pre-computed query embeddings passed in
    # Actually let's use the QA model directly on top video candidates

    hits = []
    # For each candidate video, run QA
    for vh in top_k_videos:
        if vh not in transcript_by_hash:
            continue
        segs = transcript_by_hash[vh]
        full_text, c2t = build_transcript_text_with_timestamps(segs)
        if len(full_text) < 10:
            continue

        c2t_json = json.dumps([(cs, ce, ts, te) for cs, ce, ts, te in c2t])
        start_t, end_t, score = predict_span(question, full_text, c2t_json)

        hits.append({
            'video_hash': vh,
            'start': round(start_t, 1),
            'end': round(end_t, 1),
            'score': score
        })

    hits.sort(key=lambda x: x['score'], reverse=True)
    return hits[:5]

# ── Run evaluation on train ─────────────────────────────────────
print("\nEvaluating on train...")

# Pre-compute query embeddings for stage 1
e5_model = SentenceTransformer(e5_model_path, device=DEVICE, trust_remote_code=True)
tq = train[['question_id', 'question_en']].drop_duplicates('question_id')
queries = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
qids = [row['question_id'] for _, row in tq.iterrows()]
q_emb = e5_model.encode(queries, batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True).astype('float32')
all_scores, all_indices = coarse_idx.search(q_emb, 100)
del e5_model; gc.collect(); torch.cuda.empty_cache()

# Reload QA model
qa_model = AutoModelForQuestionAnswering.from_pretrained(f'{WORK}/qa_model_best').to(DEVICE)

gt = {}
for _, row in train.iterrows():
    qid, vh = row['question_id'], extract_hash(row['video_file'])
    gt.setdefault(qid, []).append({'video_hash': vh, 'start': row['start'], 'end': row['end']})

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

# Evaluate first 200 for speed
eval_n = min(200, len(qids))
results = []
for i in tqdm(range(eval_n), desc='Train eval'):
    # Get top videos from stage 1
    video_scores = {}
    for s, j in zip(all_scores[i][:100], all_indices[i][:100]):
        if j == -1: continue
        vh = coarse_chunks[j]['video_hash']
        if vh not in video_scores or s > video_scores[vh]:
            video_scores[vh] = float(s)
    top_videos = sorted(video_scores, key=video_scores.get, reverse=True)[:10]

    hits = two_stage_predict(queries[i], coarse_idx, coarse_chunks,
                             transcript_by_hash, top_videos)
    results.append({'query_id': qids[i], 'hits': hits})

m = evaluate(gt, results)
print(f"\n=== Two-Stage QA Results (first {eval_n} queries) ===")
for k in (1, 3, 5):
    print(f"  SR@{k}={m[f'SR@{k}']:.4f}  VR@{k}={m[f'VR@{k}']:.4f}")
print(f"  AvgSR={m['AvgSR']:.4f}  AvgVR={m['AvgVR']:.4f}  FinalScore={m['FinalScore']:.4f}")

# ── Generate test submission ────────────────────────────────────
print("\nGenerating test submission...")
e5_model = SentenceTransformer(e5_model_path, device=DEVICE, trust_remote_code=True)
test_queries = [str(row['question']).lower().strip() for _, row in test.iterrows()]
test_qids = [row['query_id'] for _, row in test.iterrows()]
test_q_emb = e5_model.encode(test_queries, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True).astype('float32')
test_scores, test_indices = coarse_idx.search(test_q_emb, 100)
del e5_model; gc.collect(); torch.cuda.empty_cache()

qa_model = AutoModelForQuestionAnswering.from_pretrained(f'{WORK}/qa_model_best').to(DEVICE)

test_results = []
for i in tqdm(range(len(test_qids)), desc='Test'):
    video_scores = {}
    for s, j in zip(test_scores[i][:100], test_indices[i][:100]):
        if j == -1: continue
        vh = coarse_chunks[j]['video_hash']
        if vh not in video_scores or s > video_scores[vh]:
            video_scores[vh] = float(s)
    top_videos = sorted(video_scores, key=video_scores.get, reverse=True)[:10]

    hits = two_stage_predict(test_queries[i], coarse_idx, coarse_chunks,
                             transcript_by_hash, top_videos)

    # Fallback: if less than 5 hits, fill with coarse results
    if len(hits) < 5:
        seen = {h['video_hash'] for h in hits}
        for s, j in zip(test_scores[i], test_indices[i]):
            if j == -1: continue
            c = coarse_chunks[j]
            if c['video_hash'] not in seen:
                hits.append({'video_hash': c['video_hash'],
                            'start': c['start'], 'end': c['end'],
                            'score': float(s) * 0.1})
                seen.add(c['video_hash'])
            if len(hits) >= 5:
                break

    test_results.append({'query_id': test_qids[i], 'hits': hits[:5]})

# Build submission
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
            d[f'video_file_{rk}'] = h2f.get(list(h2f.keys())[0], 'video_unknown')
            d[f'start_{rk}'] = 0.0
            d[f'end_{rk}'] = 1.0
    rows.append(d)

cols = ['query_id']
for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
sub = pd.DataFrame(rows, columns=cols)
sub.to_csv(f'{WORK}/submission_qa_span.csv', index=False)
print(f"Saved: {WORK}/submission_qa_span.csv")
print("Done!")
