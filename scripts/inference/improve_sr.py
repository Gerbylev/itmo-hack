#!/usr/bin/env python3
"""Two-stage retrieval: E5-large-ft for video finding + fine-grained localization for SR."""

import pickle, re, os, numpy as np, pandas as pd, faiss, torch, time
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def clean_text(text):
    return text.lower().strip()

def make_chunks(segments, window, step):
    if not segments: return []
    chunks, t = [], segments[0]['start']
    while t < segments[-1]['end']:
        ws = [s for s in segments if s['end'] > t and s['start'] < t + window]
        if ws:
            chunks.append({'start': ws[0]['start'], 'end': ws[-1]['end'],
                          'text': ' '.join(clean_text(s['text']) for s in ws)})
        t += step
    return chunks

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

def print_metrics(m, name):
    print(f"\n=== {name} ===")
    for k in (1, 3, 5):
        print(f"  SR@{k}={m[f'SR@{k}']:.4f}  VR@{k}={m[f'VR@{k}']:.4f}")
    print(f"  AvgSR={m['AvgSR']:.4f}  AvgVR={m['AvgVR']:.4f}  FinalScore={m['FinalScore']:.4f}")

# ── Load data ───────────────────────────────────────────────────
print(f"Device: {DEVICE}")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)

transcript_by_hash = {}
for key, segments in transcripts.items():
    vh = extract_hash(key)
    if vh and vh not in SKIP_HASHES:
        transcript_by_hash[vh] = segments

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

# ── Stage 1: Coarse retrieval (find correct videos) ────────────
print("\n=== Stage 1: Loading E5-large fine-tuned ===")
FT_MODEL_PATH = f'{WORK}/finetuned_e5_large'
if not os.path.exists(FT_MODEL_PATH):
    FT_MODEL_PATH = 'intfloat/multilingual-e5-large'
    print(f"Fine-tuned model not found, using base: {FT_MODEL_PATH}")

model = SentenceTransformer(FT_MODEL_PATH, device=DEVICE, trust_remote_code=True)

# Build coarse chunks (30s+60s) for video-level retrieval
coarse_chunks = []
for vh, segments in transcript_by_hash.items():
    for w, s in [(30.0, 15.0), (60.0, 30.0)]:
        for ch in make_chunks(segments, w, s):
            ch['video_hash'] = vh
            coarse_chunks.append(ch)
print(f"Coarse chunks: {len(coarse_chunks)}")

coarse_emb = model.encode([ch['text'] for ch in coarse_chunks], batch_size=64,
                           show_progress_bar=True, normalize_embeddings=True,
                           convert_to_numpy=True).astype('float32')
coarse_idx = faiss.IndexFlatIP(coarse_emb.shape[1])
coarse_idx.add(coarse_emb)

# ── Stage 2: Fine-grained localization ──────────────────────────
# For each query, stage 1 gives us top videos.
# Stage 2: within each video, search with tiny chunks (10s/5s)

# Pre-build fine-grained chunks per video
print("\n=== Building fine-grained chunks per video ===")
fine_chunks_by_video = {}
for vh, segments in transcript_by_hash.items():
    fine_chunks_by_video[vh] = []
    for w, s in [(10.0, 5.0), (20.0, 10.0), (40.0, 20.0)]:
        for ch in make_chunks(segments, w, s):
            ch['video_hash'] = vh
            fine_chunks_by_video[vh].append(ch)

total_fine = sum(len(v) for v in fine_chunks_by_video.values())
print(f"Fine chunks total: {total_fine}, avg per video: {total_fine/len(fine_chunks_by_video):.0f}")

# Pre-encode all fine chunks
print("Encoding fine chunks...")
all_fine_chunks = []
fine_chunk_video_ids = []  # maps index -> video_hash
for vh, chunks in fine_chunks_by_video.items():
    for ch in chunks:
        all_fine_chunks.append(ch)
        fine_chunk_video_ids.append(vh)

fine_emb = model.encode([ch['text'] for ch in all_fine_chunks], batch_size=64,
                         show_progress_bar=True, normalize_embeddings=True,
                         convert_to_numpy=True).astype('float32')

# Build per-video FAISS indices
print("Building per-video indices...")
video_indices = {}
video_chunk_lists = {}
offset = 0
for vh, chunks in fine_chunks_by_video.items():
    n = len(chunks)
    if n == 0: continue
    sub_emb = fine_emb[offset:offset+n]
    idx = faiss.IndexFlatIP(sub_emb.shape[1])
    idx.add(sub_emb)
    video_indices[vh] = idx
    video_chunk_lists[vh] = chunks
    offset += n

# ── Two-stage retrieval function ────────────────────────────────
def two_stage_retrieve(query, model, coarse_idx, coarse_chunks,
                       video_indices, video_chunk_lists,
                       stage1_top_k=100, stage1_top_videos=10,
                       stage2_top_k=20, final_top_n=5):
    """
    Stage 1: Get top videos from coarse index
    Stage 2: Within each top video, find best fine-grained segment
    """
    qvec = model.encode([query.lower().strip()],
                        normalize_embeddings=True, convert_to_numpy=True).astype('float32')

    # Stage 1: find top videos
    scores, indices = coarse_idx.search(qvec, stage1_top_k)
    video_scores = {}
    for s, i in zip(scores[0], indices[0]):
        if i == -1: continue
        vh = coarse_chunks[i]['video_hash']
        if vh not in video_scores or s > video_scores[vh]:
            video_scores[vh] = float(s)

    top_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:stage1_top_videos]

    # Stage 2: fine-grained search within each top video
    all_hits = []
    for vh, video_score in top_videos:
        if vh not in video_indices:
            continue
        idx = video_indices[vh]
        chunks = video_chunk_lists[vh]
        k = min(stage2_top_k, idx.ntotal)
        s2_scores, s2_indices = idx.search(qvec, k)

        # Take the best segment from this video
        best_score = -1
        best_start, best_end = 0, 0

        # Collect top candidates and find best non-overlapping segments
        candidates = []
        for s, i in zip(s2_scores[0], s2_indices[0]):
            if i == -1: continue
            ch = chunks[i]
            candidates.append({
                'start': ch['start'], 'end': ch['end'],
                'score': float(s), 'video_hash': vh
            })

        if not candidates:
            continue

        # Smart merge: group overlapping, keep best
        candidates.sort(key=lambda x: x['start'])
        merged = []
        cur = candidates[0].copy()
        best_in_group = candidates[0].copy()
        for nxt in candidates[1:]:
            if nxt['start'] < cur['end']:  # overlapping
                if nxt['score'] > best_in_group['score']:
                    best_in_group = nxt.copy()
                cur['end'] = max(cur['end'], nxt['end'])
                cur['score'] = max(cur['score'], nxt['score'])
            else:
                # Shrink to best chunk in group
                bc = (best_in_group['start'] + best_in_group['end']) / 2
                bh = (best_in_group['end'] - best_in_group['start']) / 2
                merged.append({
                    'video_hash': vh,
                    'start': bc - bh, 'end': bc + bh,
                    'score': cur['score'] * video_score  # combine stage1+stage2
                })
                cur = nxt.copy()
                best_in_group = nxt.copy()
        # Last group
        bc = (best_in_group['start'] + best_in_group['end']) / 2
        bh = (best_in_group['end'] - best_in_group['start']) / 2
        merged.append({
            'video_hash': vh,
            'start': bc - bh, 'end': bc + bh,
            'score': cur['score'] * video_score
        })

        all_hits.extend(merged)

    all_hits.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate: don't return overlapping segments from same video
    final = []
    for h in all_hits:
        is_dup = False
        for f in final:
            if f['video_hash'] == h['video_hash'] and iou(f['start'], f['end'], h['start'], h['end']) > 0.3:
                is_dup = True
                break
        if not is_dup:
            final.append(h)
        if len(final) >= final_top_n:
            break

    return final

# ── Experiment with different configs ───────────────────────────
CONFIGS = [
    {'name': 'twostage_v10_t20', 'stage1_top_k': 100, 'stage1_top_videos': 10, 'stage2_top_k': 20},
    {'name': 'twostage_v15_t30', 'stage1_top_k': 150, 'stage1_top_videos': 15, 'stage2_top_k': 30},
    {'name': 'twostage_v5_t20',  'stage1_top_k': 100, 'stage1_top_videos': 5,  'stage2_top_k': 20},
    {'name': 'twostage_v10_t50', 'stage1_top_k': 100, 'stage1_top_videos': 10, 'stage2_top_k': 50},
    {'name': 'twostage_v20_t30', 'stage1_top_k': 200, 'stage1_top_videos': 20, 'stage2_top_k': 30},
]

best_score = 0
best_config = None

for cfg in CONFIGS:
    print(f"\n--- {cfg['name']} ---")
    t0 = time.time()
    results = []
    for _, row in tqdm(tq.iterrows(), total=len(tq), desc=cfg['name']):
        hits = two_stage_retrieve(
            row['question_en'], model, coarse_idx, coarse_chunks,
            video_indices, video_chunk_lists,
            cfg['stage1_top_k'], cfg['stage1_top_videos'], cfg['stage2_top_k']
        )
        results.append({'query_id': row['question_id'], 'hits': hits})

    m = evaluate(gt, results)
    print_metrics(m, cfg['name'])
    print(f"  Time: {time.time()-t0:.0f}s")

    if m['FinalScore'] > best_score:
        best_score = m['FinalScore']
        best_config = cfg

print(f"\n\nBEST CONFIG: {best_config['name']} with FinalScore={best_score:.4f}")

# ── Generate submission with best config ────────────────────────
print(f"\nGenerating submission with {best_config['name']}...")
test_results = []
for _, row in tqdm(test.iterrows(), total=len(test), desc='Test submission'):
    hits = two_stage_retrieve(
        str(row['question']), model, coarse_idx, coarse_chunks,
        video_indices, video_chunk_lists,
        best_config['stage1_top_k'], best_config['stage1_top_videos'],
        best_config['stage2_top_k']
    )
    test_results.append({'query_id': row['query_id'], 'hits': hits})

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
            d[f'video_file_{rk}'], d[f'start_{rk}'], d[f'end_{rk}'] = '', 0.0, 0.0
    rows.append(d)
cols = ['query_id']
for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']
sub = pd.DataFrame(rows, columns=cols)
sub.to_csv(f'{WORK}/submission_twostage.csv', index=False)
print(f"Saved: {WORK}/submission_twostage.csv")
print("Done!")
