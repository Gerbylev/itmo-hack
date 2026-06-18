"""
Solution: E5-large fine-tuned + 3-scale chunks + shrink=0.95
Train FinalScore: 0.5625
Kaggle: 0.471
"""
import pickle, re, numpy as np, pandas as pd, faiss, torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = '/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_HASHES = {'7d49c038'}

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

def merge(cands, gap=10, shrink=0.95):
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
                bc = (best['start'] + best['end']) / 2
                bh = (best['end'] - best['start']) / 2
                ch = (cur['end'] - cur['start']) / 2
                nh = ch * (1 - shrink) + bh * shrink
                cur['start'] = max(cur['start'], bc - nh)
                cur['end'] = min(cur['end'], bc + nh)
                merged.append(cur)
                cur, best = nxt.copy(), nxt.copy()
        bc = (best['start'] + best['end']) / 2
        bh = (best['end'] - best['start']) / 2
        ch = (cur['end'] - cur['start']) / 2
        nh = ch * (1 - shrink) + bh * shrink
        cur['start'] = max(cur['start'], bc - nh)
        cur['end'] = min(cur['end'], bc + nh)
        merged.append(cur)
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged

# Load data
with open(f'{BASE}/transcripts.pkl', 'rb') as f:
    transcripts = pickle.load(f)

h2f = {}
for p in pd.read_csv(f'{BASE}/video_files.csv')['video_path']:
    h = extract_hash(p)
    if h: h2f[h] = re.sub(r'\.\w+$', '', p.split('/')[-1])

test = pd.read_csv(f'{BASE}/test/test.csv')

# Build chunks: 30s/15s + 60s/30s + 90s/45s
all_chunks = []
for key, segs in tqdm(transcripts.items()):
    vh = extract_hash(key)
    if not vh or vh in SKIP_HASHES: continue
    for w, s in [(30, 15), (60, 30), (90, 45)]:
        for ch in make_chunks(segs, w, s):
            ch['video_hash'] = vh
            all_chunks.append(ch)
print(f'Chunks: {len(all_chunks)}')

# Load fine-tuned E5-large
model = SentenceTransformer('olegGerbylev/e5-large-video-retrieval-ft-v2',
                            device=DEVICE, trust_remote_code=True)

# Encode chunks
emb = model.encode([ch['text'] for ch in all_chunks], batch_size=64,
                    show_progress_bar=True, normalize_embeddings=True,
                    convert_to_numpy=True).astype('float32')

# FAISS index
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# Inference
results = []
for _, row in tqdm(test.iterrows(), total=len(test)):
    q = str(row['question']).lower().strip()
    qv = model.encode([q], normalize_embeddings=True,
                      convert_to_numpy=True).astype('float32')
    scores, ids = index.search(qv, 100)
    cands = [{'video_hash': all_chunks[j]['video_hash'],
              'start': all_chunks[j]['start'],
              'end': all_chunks[j]['end'],
              'score': float(s)}
             for s, j in zip(scores[0], ids[0]) if j != -1]
    hits = merge(cands, gap=10, shrink=0.95)[:5]
    results.append({'query_id': row['query_id'], 'hits': hits})

# Build submission
fallback = list(h2f.values())[0]
rows = []
for r in results:
    d = {'query_id': r['query_id']}
    for rk in range(1, 6):
        if rk <= len(r['hits']):
            h = r['hits'][rk - 1]
            d[f'video_file_{rk}'] = h2f.get(h['video_hash'], fallback)
            d[f'start_{rk}'] = round(h['start'], 1)
            d[f'end_{rk}'] = round(h['end'], 1)
        else:
            d[f'video_file_{rk}'] = fallback
            d[f'start_{rk}'] = 0.0
            d[f'end_{rk}'] = 1.0
    rows.append(d)

cols = ['query_id']
for rk in range(1, 6):
    cols += [f'video_file_{rk}', f'start_{rk}', f'end_{rk}']

sub = pd.DataFrame(rows, columns=cols)
sub.to_csv('/kaggle/working/submission.csv', index=False)
print(f'Shape: {sub.shape}')
sub.head()
