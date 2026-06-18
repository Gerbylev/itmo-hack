#!/usr/bin/env python3
"""
Radical SR improvement: per-segment similarity + max-sum subsequence.
No fixed chunks for temporal localization.
"""
import pickle, re, os, numpy as np, pandas as pd, faiss, torch, gc, time
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)

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

def print_metrics(m, name):
    print(f"\n=== {name} ===")
    for k in (1,3,5):
        print(f"  SR@{k}={m[f'SR@{k}']:.4f}  VR@{k}={m[f'VR@{k}']:.4f}")
    print(f"  AvgSR={m['AvgSR']:.4f}  AvgVR={m['AvgVR']:.4f}  FinalScore={m['FinalScore']:.4f}")

def make_submission(results, h2f, path):
    fallback = list(h2f.values())[0]
    rows = []
    for r in results:
        d = {'query_id': r['query_id']}
        for rk in range(1, 6):
            if rk <= len(r['hits']):
                h = r['hits'][rk-1]
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
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    print(f"  Saved: {path}")

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
    h = extract_hash(p); fn = re.sub(r'\.\w+$', '', p.split('/')[-1])
    if h: h2f[h] = fn

gt = {}
for _, row in train.iterrows():
    qid, vh = row['question_id'], extract_hash(row['video_file'])
    gt.setdefault(qid, []).append({'video_hash': vh, 'start': row['start'], 'end': row['end']})
tq = train[['question_id', 'question_en']].drop_duplicates('question_id')

# ── Stage 1: Video retrieval with E5-ft (coarse chunks) ────────
print("\n=== Stage 1: Video Retrieval ===")
model = SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2', device=DEVICE, trust_remote_code=True)

coarse_chunks = []
for vh, segs in transcript_by_hash.items():
    for w, s in [(30,15),(60,30),(90,45)]:
        for ch in make_chunks(segs, w, s):
            ch['video_hash'] = vh
            coarse_chunks.append(ch)
print(f"Coarse chunks: {len(coarse_chunks)}")

coarse_emb = model.encode([c['text'] for c in coarse_chunks], batch_size=64,
                           show_progress_bar=True, normalize_embeddings=True,
                           convert_to_numpy=True).astype('float32')
coarse_idx = faiss.IndexFlatIP(coarse_emb.shape[1])
coarse_idx.add(coarse_emb)

def get_top_videos(query_emb, top_k_chunks=100, top_n_videos=10):
    scores, indices = coarse_idx.search(query_emb, top_k_chunks)
    video_scores = {}
    for s, j in zip(scores[0], indices[0]):
        if j == -1: continue
        vh = coarse_chunks[j]['video_hash']
        if vh not in video_scores or s > video_scores[vh]:
            video_scores[vh] = float(s)
    return sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_videos]

# ── Pre-encode all individual segments per video ────────────────
print("\n=== Pre-encoding individual Whisper segments ===")
all_seg_texts = []
seg_meta = []  # (video_hash, start, end, idx_in_video)
for vh, segs in transcript_by_hash.items():
    for i, seg in enumerate(segs):
        t = seg['text'].strip()
        if not t: continue
        all_seg_texts.append(t.lower())
        seg_meta.append({'video_hash': vh, 'start': seg['start'], 'end': seg['end'], 'idx': i})

print(f"Total individual segments: {len(all_seg_texts)}")
seg_emb = model.encode(all_seg_texts, batch_size=128, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype('float32')

# Build per-video segment indices
seg_by_video = {}
for i, meta in enumerate(seg_meta):
    vh = meta['video_hash']
    if vh not in seg_by_video:
        seg_by_video[vh] = {'embs': [], 'starts': [], 'ends': []}
    seg_by_video[vh]['embs'].append(seg_emb[i])
    seg_by_video[vh]['starts'].append(meta['start'])
    seg_by_video[vh]['ends'].append(meta['end'])

for vh in seg_by_video:
    seg_by_video[vh]['embs'] = np.array(seg_by_video[vh]['embs'])

print(f"Videos with segments: {len(seg_by_video)}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Max-Sum Subsequence (basic)
# ══════════════════════════════════════════════════════════════
def find_best_segment_maxsum(query_emb, vh, min_segs=2, max_segs=50):
    """Find contiguous segment with maximum similarity sum."""
    if vh not in seg_by_video:
        return None
    data = seg_by_video[vh]
    embs = data['embs']
    starts = data['starts']
    ends = data['ends']
    n = len(embs)
    if n == 0: return None

    # Compute per-segment similarity
    sims = embs @ query_emb.flatten()

    # Subtract median to make "neutral" segments ~0
    median_sim = np.median(sims)
    adjusted = sims - median_sim

    # Find max-sum contiguous subarray with length constraints
    best_score = -float('inf')
    best_i, best_j = 0, min(n-1, min_segs-1)

    for i in range(n):
        cumsum = 0
        for j in range(i, min(i + max_segs, n)):
            cumsum += adjusted[j]
            seg_len = j - i + 1
            if seg_len >= min_segs and cumsum > best_score:
                best_score = cumsum
                best_i, best_j = i, j

    return {
        'start': starts[best_i],
        'end': ends[best_j],
        'score': float(best_score),
    }

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Max-Sum with smoothing
# ══════════════════════════════════════════════════════════════
def find_best_segment_smooth(query_emb, vh, kernel=5, min_segs=2, max_segs=50):
    if vh not in seg_by_video:
        return None
    data = seg_by_video[vh]
    embs = data['embs']
    starts = data['starts']
    ends = data['ends']
    n = len(embs)
    if n == 0: return None

    sims = embs @ query_emb.flatten()

    # Smooth with moving average
    kernel = min(kernel, n)
    smoothed = np.convolve(sims, np.ones(kernel)/kernel, mode='same')

    median_sim = np.median(smoothed)
    adjusted = smoothed - median_sim

    best_score = -float('inf')
    best_i, best_j = 0, min(n-1, min_segs-1)

    for i in range(n):
        cumsum = 0
        for j in range(i, min(i + max_segs, n)):
            cumsum += adjusted[j]
            if (j - i + 1) >= min_segs and cumsum > best_score:
                best_score = cumsum
                best_i, best_j = i, j

    return {'start': starts[best_i], 'end': ends[best_j], 'score': float(best_score)}

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Peak detection — find highest peak region
# ══════════════════════════════════════════════════════════════
def find_best_segment_peak(query_emb, vh, threshold_pct=75, min_segs=2):
    if vh not in seg_by_video:
        return None
    data = seg_by_video[vh]
    embs = data['embs']
    starts = data['starts']
    ends = data['ends']
    n = len(embs)
    if n == 0: return None

    sims = embs @ query_emb.flatten()

    # Smooth
    kernel = min(5, n)
    smoothed = np.convolve(sims, np.ones(kernel)/kernel, mode='same')

    # Threshold: segments above Nth percentile
    thresh = np.percentile(smoothed, threshold_pct)
    above = smoothed >= thresh

    # Find contiguous runs above threshold
    runs = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            score = float(np.sum(smoothed[i:j]))
            runs.append((i, j-1, score))
            i = j
        else:
            i += 1

    if not runs:
        # Fallback: take peak ± 5 segments
        peak = int(np.argmax(smoothed))
        i = max(0, peak - 5)
        j = min(n-1, peak + 5)
        return {'start': starts[i], 'end': ends[j], 'score': float(smoothed[peak])}

    # Take best run
    best_run = max(runs, key=lambda x: x[2])
    return {'start': starts[best_run[0]], 'end': ends[best_run[1]], 'score': best_run[2]}

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Hybrid — coarse chunk score × fine segment score
# ══════════════════════════════════════════════════════════════
def find_best_segment_hybrid(query_emb, vh, video_score, method='maxsum'):
    if method == 'maxsum':
        result = find_best_segment_maxsum(query_emb, vh)
    elif method == 'smooth':
        result = find_best_segment_smooth(query_emb, vh)
    elif method == 'peak':
        result = find_best_segment_peak(query_emb, vh)
    else:
        result = find_best_segment_maxsum(query_emb, vh)

    if result is None:
        return None
    result['score'] = result['score'] + video_score * 10  # combine
    result['video_hash'] = vh
    return result

# ── Run all experiments ─────────────────────────────────────────
def run_experiment(name, method, queries, qids, gt_dict=None, is_test=False):
    print(f"\n--- {name} ---")
    t0 = time.time()
    results = []

    for i in tqdm(range(len(queries)), desc=name):
        q = queries[i]
        q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        top_videos = get_top_videos(q_emb, top_k_chunks=100, top_n_videos=10)

        hits = []
        for vh, vs in top_videos:
            result = find_best_segment_hybrid(q_emb, vh, vs, method=method)
            if result:
                hits.append(result)

        hits.sort(key=lambda x: x['score'], reverse=True)
        results.append({'query_id': qids[i], 'hits': hits[:5]})

    elapsed = time.time() - t0
    if gt_dict:
        m = evaluate(gt_dict, results)
        print_metrics(m, name)
        print(f"  Time: {elapsed:.0f}s")
        return results, m
    return results, None

# Encode queries
print("\nEncoding train queries...")
train_queries = [row['question_en'].lower().strip() for _, row in tq.iterrows()]
train_qids = [row['question_id'] for _, row in tq.iterrows()]

print("Encoding test queries...")
test_queries = [str(row['question']).lower().strip() for _, row in test.iterrows()]
test_qids = [row['query_id'] for _, row in test.iterrows()]

# Run experiments on train
experiments = [
    ('maxsum_basic', 'maxsum'),
    ('smooth_k5', 'smooth'),
    ('peak_p75', 'peak'),
]

best_method = None
best_score = 0

for exp_name, method in experiments:
    _, m = run_experiment(exp_name, method, train_queries, train_qids, gt)
    if m and m['FinalScore'] > best_score:
        best_score = m['FinalScore']
        best_method = method

print(f"\nBest method: {best_method} with FinalScore={best_score:.4f}")

# ── Generate test submission with best method ───────────────────
print(f"\nGenerating test submission with {best_method}...")
test_results, _ = run_experiment(f'test_{best_method}', best_method, test_queries, test_qids)
make_submission(test_results, h2f, f'{WORK}/submission_radical_{best_method}.csv')

# Also generate for all methods
for exp_name, method in experiments:
    test_results, _ = run_experiment(f'test_{exp_name}', method, test_queries, test_qids)
    make_submission(test_results, h2f, f'{WORK}/submission_radical_{exp_name}.csv')

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 5: Cross-encoder reranking on top segments
# ══════════════════════════════════════════════════════════════
print("\n=== Cross-encoder reranking ===")
del model; gc.collect(); torch.cuda.empty_cache()

cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=DEVICE)

def rerank_with_cross_encoder(query, vh, candidates_start_end, max_pairs=20):
    """Rerank candidate segments using cross-encoder."""
    if vh not in transcript_by_hash:
        return candidates_start_end
    segs = transcript_by_hash[vh]

    pairs = []
    for cand in candidates_start_end[:max_pairs]:
        # Get transcript text for this candidate
        ws = [s for s in segs if s['end'] > cand['start'] and s['start'] < cand['end']]
        text = ' '.join(s['text'].strip() for s in ws)
        if text:
            pairs.append((query, text))
        else:
            pairs.append((query, "empty"))

    if not pairs:
        return candidates_start_end

    scores = cross_model.predict(pairs)
    for i, s in enumerate(scores):
        if i < len(candidates_start_end):
            candidates_start_end[i]['cross_score'] = float(s)

    candidates_start_end.sort(key=lambda x: x.get('cross_score', -999), reverse=True)
    return candidates_start_end

# Reload bi-encoder for stage 1
model = SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2', device=DEVICE, trust_remote_code=True)

print("Running cross-encoder reranking on train...")
results_ce = []
for i in tqdm(range(len(train_queries)), desc='Cross-encoder'):
    q = train_queries[i]
    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    top_videos = get_top_videos(q_emb, top_k_chunks=100, top_n_videos=10)

    hits = []
    for vh, vs in top_videos:
        # Get multiple candidate segments per video
        for method in ['maxsum', 'smooth', 'peak']:
            if method == 'maxsum':
                r = find_best_segment_maxsum(q_emb, vh)
            elif method == 'smooth':
                r = find_best_segment_smooth(q_emb, vh)
            else:
                r = find_best_segment_peak(q_emb, vh)
            if r:
                r['video_hash'] = vh
                r['video_score'] = vs
                hits.append(r)

    # Deduplicate by video
    seen = set()
    unique_hits = []
    for h in hits:
        key = (h['video_hash'], round(h['start'],0), round(h['end'],0))
        if key not in seen:
            seen.add(key)
            unique_hits.append(h)

    # Rerank with cross-encoder (group by video)
    by_video = {}
    for h in unique_hits:
        by_video.setdefault(h['video_hash'], []).append(h)

    final_hits = []
    for vh, cands in by_video.items():
        reranked = rerank_with_cross_encoder(q, vh, cands)
        if reranked:
            best = reranked[0]
            best['final_score'] = best.get('cross_score', 0) + best.get('video_score', 0)
            final_hits.append(best)

    final_hits.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    results_ce.append({'query_id': train_qids[i], 'hits': final_hits[:5]})

m_ce = evaluate(gt, results_ce)
print_metrics(m_ce, 'cross_encoder_reranked')

# Generate test submission with cross-encoder
print("\nGenerating test submission with cross-encoder...")
test_results_ce = []
for i in tqdm(range(len(test_queries)), desc='CE Test'):
    q = test_queries[i]
    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    top_videos = get_top_videos(q_emb, top_k_chunks=100, top_n_videos=10)

    hits = []
    for vh, vs in top_videos:
        for method in ['maxsum', 'smooth', 'peak']:
            if method == 'maxsum':
                r = find_best_segment_maxsum(q_emb, vh)
            elif method == 'smooth':
                r = find_best_segment_smooth(q_emb, vh)
            else:
                r = find_best_segment_peak(q_emb, vh)
            if r:
                r['video_hash'] = vh
                r['video_score'] = vs
                hits.append(r)

    seen = set()
    unique_hits = []
    for h in hits:
        key = (h['video_hash'], round(h['start'],0), round(h['end'],0))
        if key not in seen:
            seen.add(key)
            unique_hits.append(h)

    by_video = {}
    for h in unique_hits:
        by_video.setdefault(h['video_hash'], []).append(h)

    final_hits = []
    for vh, cands in by_video.items():
        reranked = rerank_with_cross_encoder(q, vh, cands)
        if reranked:
            best = reranked[0]
            best['final_score'] = best.get('cross_score', 0) + best.get('video_score', 0)
            final_hits.append(best)

    final_hits.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    test_results_ce.append({'query_id': test_qids[i], 'hits': final_hits[:5]})

make_submission(test_results_ce, h2f, f'{WORK}/submission_radical_cross_encoder.csv')

print("\n" + "="*60)
print("ALL EXPERIMENTS DONE")
print("="*60)
print("Submissions in /root/output/submission_radical_*.csv")
