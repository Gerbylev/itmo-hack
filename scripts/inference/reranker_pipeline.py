#!/usr/bin/env python3
"""Two-stage pipeline: BGE-M3 FAISS retrieval + Qwen2.5-VL reranking for video moment localization.
Optimized version: fewer candidates, fewer frames for speed."""

import pickle, re, os, json, subprocess, sys, gc, glob, shutil
import numpy as np, pandas as pd, faiss, torch, time
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = '/root/data/video-rag'
WORK = '/root/output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}
os.makedirs(WORK, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────
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

def get_transcript_for_segment(transcripts, video_hash, start, end):
    """Get transcript text for a video segment."""
    for key, segments in transcripts.items():
        vh = extract_hash(key)
        if vh == video_hash:
            relevant = [s for s in segments if s['end'] > start and s['start'] < end]
            if relevant:
                return ' '.join(clean_text(s['text']) for s in relevant)
    return ''

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
              'end': chunks[i]['end'], 'text': chunks[i]['text'], 'score': float(s)}
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

# ── Frame extraction ─────────────────────────────────────────────
def extract_frames(video_path, start, end, n_frames=3, out_dir='/tmp/frames'):
    """Extract n_frames evenly spaced frames from video between start and end."""
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    timestamps = np.linspace(start, end, n_frames + 2)[1:-1]

    for i, ts in enumerate(timestamps):
        out_path = f"{out_dir}/frame_{i:03d}.jpg"
        cmd = [
            'ffmpeg', '-y', '-ss', str(round(ts, 2)), '-i', video_path,
            '-vframes', '1', '-q:v', '3', '-s', '336x336', out_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                frames.append(out_path)
        except:
            pass
    return frames

# ── Video file resolution ────────────────────────────────────────
def build_video_hash_map(video_dir):
    h2path = {}
    for f in glob.glob(os.path.join(video_dir, '*')):
        if f.endswith(('.mp4', '.mkv', '.webm', '.avi')):
            h = extract_hash(f)
            if h: h2path[h] = f
    return h2path

# ── Qwen VL Reranker ────────────────────────────────────────────
class QwenReranker:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Qwen model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    @torch.no_grad()
    def score_candidate(self, query, transcript_text, frame_paths):
        """Score how well frames+transcript match the query. Returns float 0-10."""
        if not frame_paths:
            return 5.0

        content = []
        for fp in frame_paths[:3]:
            content.append({"type": "image", "image": f"file://{fp}"})

        prompt_text = (
            f"Query: {query}\n"
            f"Transcript: {transcript_text[:300]}\n\n"
            f"How well do these frames and transcript match the query? "
            f"Reply with ONLY a number 0-10."
        )
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(generated, skip_special_tokens=True).strip()

            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                return min(max(float(match.group(1)), 0.0), 10.0)
            return 5.0
        except Exception as e:
            return 5.0

    def rerank_batch(self, query, candidates, h2path, transcripts, n_frames=3):
        """Score all candidates for a query."""
        results = []
        for cand in candidates:
            vh = cand['video_hash']
            video_path = h2path.get(vh)

            if video_path is None:
                results.append({**cand, 'vlm_score': 5.0})
                continue

            frame_dir = f'/tmp/frames/{vh}_{int(cand["start"])}_{int(cand["end"])}'
            frame_paths = extract_frames(video_path, cand['start'], cand['end'],
                                        n_frames=n_frames, out_dir=frame_dir)
            transcript_text = cand.get('text', '')
            if not transcript_text:
                transcript_text = get_transcript_for_segment(transcripts, vh, cand['start'], cand['end'])

            vlm_score = self.score_candidate(query, transcript_text, frame_paths)
            results.append({**cand, 'vlm_score': vlm_score})

            try: shutil.rmtree(frame_dir, ignore_errors=True)
            except: pass

        return results


# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_total = time.time()

    print("=" * 60)
    print("STAGE 1: BGE-M3 + FAISS Retrieval")
    print("=" * 60)

    # Load data
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
    print(f"Train queries: {len(tq)}, Test queries: {len(test)}")

    # Build chunks
    chunk_config = [(30.0, 15.0), (60.0, 30.0)]
    chunks = build_chunks(chunk_config, transcripts)
    print(f"Built {len(chunks)} chunks")

    # Load BGE-M3
    print("Loading BGE-M3...")
    model = SentenceTransformer('BAAI/bge-m3', device=DEVICE, trust_remote_code=True)

    print("Encoding chunks...")
    texts = [ch['text'] for ch in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype('float32')

    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    TOP_K_FAISS = 100
    TOP_N_RERANK = 7  # candidates per query for VLM reranking
    GAP = 10
    SHRINK = 0.3

    # Stage 1: Train
    print(f"\nStage 1: Train retrieval (top-{TOP_N_RERANK})...")
    stage1_train = []
    for _, row in tqdm(tq.iterrows(), total=len(tq)):
        q = row['question_en'].lower().strip()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qv, idx, chunks, TOP_K_FAISS, TOP_N_RERANK, GAP, SHRINK)
        stage1_train.append({'query_id': row['question_id'], 'query': row['question_en'], 'hits': hits})

    m1 = evaluate(gt, stage1_train)
    print(f"\nSTAGE 1 BASELINE:")
    print(f"  SR@1={m1['SR@1']:.4f} SR@3={m1['SR@3']:.4f} SR@5={m1['SR@5']:.4f}")
    print(f"  VR@1={m1['VR@1']:.4f} VR@3={m1['VR@3']:.4f} VR@5={m1['VR@5']:.4f}")
    print(f"  AvgSR={m1['AvgSR']:.4f} AvgVR={m1['AvgVR']:.4f} FinalScore={m1['FinalScore']:.4f}")

    # Stage 1: Test
    print(f"\nStage 1: Test retrieval...")
    stage1_test = []
    for _, row in tqdm(test.iterrows(), total=len(test)):
        q = str(row['question']).lower().strip()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        hits = retrieve(qv, idx, chunks, TOP_K_FAISS, TOP_N_RERANK, GAP, SHRINK)
        stage1_test.append({'query_id': row['query_id'], 'query': str(row['question']), 'hits': hits})

    # Save baseline submission
    cols = ['query_id']
    for rk in range(1, 6): cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']

    rows_base = []
    for r in stage1_test:
        d = {'query_id': r['query_id']}
        for rk in range(1, 6):
            if rk <= len(r['hits']):
                h = r['hits'][rk-1]
                d[f'video_file_{rk}'] = h2f.get(h['video_hash'], h['video_hash'])
                d[f'start_{rk}'] = round(h['start'], 1)
                d[f'end_{rk}'] = round(h['end'], 1)
            else:
                d[f'video_file_{rk}'], d[f'start_{rk}'], d[f'end_{rk}'] = '', 0.0, 0.0
        rows_base.append(d)
    pd.DataFrame(rows_base, columns=cols).to_csv(f'{WORK}/submission_baseline_v4.csv', index=False)
    print(f"Baseline saved to {WORK}/submission_baseline_v4.csv")

    # Free BGE
    del model, emb, idx
    gc.collect(); torch.cuda.empty_cache()

    # ── STAGE 2: Qwen VL Reranking ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Qwen VL Reranking")
    print("=" * 60)

    video_dir = f'{BASE}/videos'
    h2path = build_video_hash_map(video_dir)
    print(f"Found {len(h2path)} video files")

    reranker = QwenReranker("Qwen/Qwen2.5-VL-3B-Instruct")

    # Rerank train
    N_FRAMES = 3
    print(f"\nReranking train ({len(stage1_train)} queries, {TOP_N_RERANK} cands, {N_FRAMES} frames)...")
    t2 = time.time()
    train_vlm_scores = []
    for i, r in enumerate(tqdm(stage1_train)):
        scored = reranker.rerank_batch(r['query'], r['hits'], h2path, transcripts, n_frames=N_FRAMES)
        train_vlm_scores.append({'query_id': r['query_id'], 'scored_hits': scored})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t2
            rate = elapsed / (i + 1)
            eta = rate * (len(stage1_train) - i - 1)
            print(f"  [{i+1}/{len(stage1_train)}] {rate:.1f}s/query, eta={eta/60:.0f}min")

    print(f"Train reranking done in {(time.time()-t2)/60:.1f}min")

    # Tune weights
    print("\nTuning combination weights...")
    best_score = 0
    best_alpha, best_beta = 1.0, 0.0
    best_metrics = None

    for alpha in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        for beta in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if alpha == 0 and beta == 0: continue
            tuned = []
            for r in train_vlm_scores:
                hits = [{**h} for h in r['scored_hits']]
                for h in hits:
                    h['combined'] = h['score'] * alpha + h.get('vlm_score', 5.0) * beta
                hits.sort(key=lambda x: x['combined'], reverse=True)
                tuned.append({'query_id': r['query_id'], 'hits': hits[:5]})
            m_t = evaluate(gt, tuned)
            if m_t['FinalScore'] > best_score:
                best_score = m_t['FinalScore']
                best_alpha, best_beta = alpha, beta
                best_metrics = m_t
                print(f"  alpha={alpha}, beta={beta} -> FS={m_t['FinalScore']:.4f} "
                      f"(SR={m_t['AvgSR']:.4f}, VR={m_t['AvgVR']:.4f})")

    print(f"\nBest: alpha={best_alpha}, beta={best_beta}, FinalScore={best_score:.4f}")
    if best_metrics:
        print(f"  SR@1={best_metrics['SR@1']:.4f} SR@3={best_metrics['SR@3']:.4f} SR@5={best_metrics['SR@5']:.4f}")
        print(f"  VR@1={best_metrics['VR@1']:.4f} VR@3={best_metrics['VR@3']:.4f} VR@5={best_metrics['VR@5']:.4f}")

    print(f"\nIMPROVEMENT vs Stage 1:")
    print(f"  FinalScore: {m1['FinalScore']:.4f} -> {best_score:.4f} ({best_score-m1['FinalScore']:+.4f})")
    if best_metrics:
        print(f"  AvgSR: {m1['AvgSR']:.4f} -> {best_metrics['AvgSR']:.4f}")
        print(f"  AvgVR: {m1['AvgVR']:.4f} -> {best_metrics['AvgVR']:.4f}")

    # Rerank test
    print(f"\nReranking test ({len(stage1_test)} queries)...")
    t3 = time.time()
    reranked_test = []
    for i, r in enumerate(tqdm(stage1_test)):
        scored = reranker.rerank_batch(r['query'], r['hits'], h2path, transcripts, n_frames=N_FRAMES)
        for h in scored:
            h['combined'] = h['score'] * best_alpha + h.get('vlm_score', 5.0) * best_beta
        scored.sort(key=lambda x: x['combined'], reverse=True)
        reranked_test.append({'query_id': r['query_id'], 'hits': scored[:5]})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t3
            rate = elapsed / (i + 1)
            eta = rate * (len(stage1_test) - i - 1)
            print(f"  [{i+1}/{len(stage1_test)}] eta={eta/60:.0f}min")

    print(f"Test reranking done in {(time.time()-t3)/60:.1f}min")

    # Generate submission
    rows = []
    for r in reranked_test:
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

    sub_path = f'{WORK}/submission_reranked.csv'
    pd.DataFrame(rows, columns=cols).to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")

    total = time.time() - t_total
    print(f"\nTOTAL TIME: {total/60:.1f}min")
    print("DONE!")
