"""
Kaggle Submission Notebook - Fine-tuned BGE-M3 for Video Fragment Retrieval
============================================================================
Instructions:
1. Upload fine-tuned model to HuggingFace (see below)
2. Create a new Kaggle notebook for this competition
3. Enable GPU (T4 is enough)
4. Paste this code and run

To upload model to HF:
    pip install huggingface_hub
    huggingface-cli login
    cd /home/oleg/hack/finetune_results
    tar xzf finetuned_bge_m3.tar.gz
    huggingface-cli upload YOUR_USERNAME/bge-m3-video-retrieval-ft finetuned_bge_m3/ . --repo-type model
"""

# Cell 1: Install deps
# !pip install faiss-cpu sentence-transformers

# Cell 2: Imports and config
import pickle
import re
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = '/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag'
WORK = '/kaggle/working'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}

# ======== CHANGE THIS to your HF model name ========
MODEL_NAME = 'olegGerbylev/bge-m3-video-retrieval-ft'
# ====================================================

# Cell 3: Helpers
def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

def clean_text(text):
    return text.lower().strip()

def make_chunks(segments, window, step):
    if not segments:
        return []
    chunks, total_end, t_start = [], segments[-1]['end'], segments[0]['start']
    while t_start < total_end:
        t_end = t_start + window
        ws = [s for s in segments if s['end'] > t_start and s['start'] < t_end]
        if ws:
            chunks.append({
                'start': ws[0]['start'],
                'end': ws[-1]['end'],
                'text': ' '.join(clean_text(s['text']) for s in ws),
            })
        t_start += step
    return chunks

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

def retrieve(qvec, index, chunks, top_k=100, top_n=5, gap=10.0, shrink=0.7):
    scores, indices = index.search(qvec, top_k)
    cands = [{'video_hash': chunks[i]['video_hash'], 'start': chunks[i]['start'],
              'end': chunks[i]['end'], 'score': float(s)}
             for s, i in zip(scores[0], indices[0]) if i != -1]
    return merge_chunks_v2(cands, gap, shrink)[:top_n]

# Cell 4: Load data
print("Loading transcripts...")
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)

video_files = pd.read_csv(f'{BASE}/video_files.csv')
hash_to_filename = {}
for path in video_files['video_path']:
    h = extract_hash(path)
    filename = re.sub(r'\.\w+$', '', path.split('/')[-1])
    if h:
        hash_to_filename[h] = filename

test = pd.read_csv(f'{BASE}/test/test.csv')
print(f"Test queries: {len(test)}")

# Cell 5: Build chunks (30s + 60s windows)
print("Building chunks...")
all_chunks = []
for key, segments in tqdm(transcripts.items()):
    video_hash = extract_hash(key)
    if video_hash is None or video_hash in SKIP_HASHES:
        continue
    for w, s in [(30.0, 15.0), (60.0, 30.0)]:
        for ch in make_chunks(segments, w, s):
            ch['video_hash'] = video_hash
            all_chunks.append(ch)
print(f"Total chunks: {len(all_chunks)}")

# Cell 6: Load model and encode
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)

texts = [ch['text'] for ch in all_chunks]
print(f"Encoding {len(texts)} chunks on {DEVICE}...")
embeddings = model.encode(
    texts, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
).astype('float32')

print(f"Embeddings shape: {embeddings.shape}")

# Cell 7: Build FAISS index
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"Index: {index.ntotal} vectors")

# Cell 8: Run inference
TOP_K = 100
MERGE_GAP = 10.0
SHRINK = 0.7

print(f"\nRunning inference on {len(test)} test queries...")
results = []
for _, row in tqdm(test.iterrows(), total=len(test)):
    question = str(row['question']).lower().strip()
    qvec = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype('float32')
    hits = retrieve(qvec, index, all_chunks, TOP_K, 5, MERGE_GAP, SHRINK)
    results.append({'query_id': row['query_id'], 'hits': hits})

print(f"Inference done. Results: {len(results)}")

# Cell 9: Build submission
rows = []
for r in results:
    row_d = {'query_id': r['query_id']}
    for rank in range(1, 6):
        if rank <= len(r['hits']):
            h = r['hits'][rank - 1]
            fname = hash_to_filename.get(h['video_hash'], h['video_hash'])
            row_d[f'video_file_{rank}'] = fname
            row_d[f'start_{rank}'] = round(h['start'], 1)
            row_d[f'end_{rank}'] = round(h['end'], 1)
        else:
            row_d[f'video_file_{rank}'] = ''
            row_d[f'start_{rank}'] = 0.0
            row_d[f'end_{rank}'] = 0.0
    rows.append(row_d)

cols = ['query_id']
for rank in range(1, 6):
    cols += [f'video_file_{rank}', f'start_{rank}', f'end_{rank}']

submission = pd.DataFrame(rows, columns=cols)
submission.to_csv(f'{WORK}/submission.csv', index=False)

print(f"Submission shape: {submission.shape}")
print(submission.head(3).to_string())
print(f"\nSaved: {WORK}/submission.csv")

# Validate
assert submission['query_id'].nunique() == len(test), "Missing query_ids!"
assert not submission['video_file_1'].isna().any(), "NaN in video_file_1!"
print("Validation OK. Ready to submit!")
