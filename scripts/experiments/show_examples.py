import pandas as pd, pickle, re

BASE = "/root/data/video-rag"
train = pd.read_csv(f"{BASE}/train/train_qa.csv")
with open(f"{BASE}/transcripts.pkl", "rb") as f:
    transcripts = pickle.load(f)

def extract_hash(path):
    m = re.search(r'_([a-f0-9]+)[\.\w]*$', str(path))
    return m.group(1) if m else None

th = {}
for key, segs in transcripts.items():
    vh = extract_hash(key)
    if vh: th[vh] = segs

train["duration"] = train["end"] - train["start"]

samples_idx = []
samples_idx.append(train[(train["duration"]>15)&(train["duration"]<30)].sample(1, random_state=42).index[0])
samples_idx.append(train[(train["duration"]>50)&(train["duration"]<70)].sample(1, random_state=42).index[0])
samples_idx.append(train[(train["duration"]>100)&(train["duration"]<140)].sample(1, random_state=42).index[0])
samples_idx.append(train[(train["duration"]>5)&(train["duration"]<15)].sample(1, random_state=42).index[0])

for i, idx in enumerate(samples_idx):
    s = train.iloc[idx]
    vh = extract_hash(s.video_file)
    segs = th.get(vh, [])

    gt_segs = [seg for seg in segs if seg["end"] > s.start and seg["start"] < s.end]
    gt_text = " ".join(seg["text"].strip() for seg in gt_segs)

    before_segs = [seg for seg in segs if seg["end"] <= s.start and seg["start"] >= s.start - 30]
    before_text = " ".join(seg["text"].strip() for seg in before_segs)
    if not before_text:
        before_text = "(empty)"

    after_segs = [seg for seg in segs if seg["start"] >= s.end and seg["start"] < s.end + 30]
    after_text = " ".join(seg["text"].strip() for seg in after_segs)
    if not after_text:
        after_text = "(empty)"

    sep = "=" * 80
    print(sep)
    dur = int(s.duration)
    print(f"EXAMPLE {i+1} | Topic: {s.topic} | Duration: {dur}s")
    print(sep)
    print(f"Video: {s.video_file} | Hash: {vh}")
    print(f"GT segment: {s.start}s - {s.end}s")
    print()
    print("QUESTION (EN):")
    print(f"  {s.question_en}")
    print("QUESTION (RU):")
    print(f"  {s.question_ru}")
    print()
    print("ANSWER (first 300 chars):")
    ans = s.answer_en[:300]
    print(f"  {ans}...")
    print()
    start_before = int(s.start - 30)
    start_gt = int(s.start)
    end_gt = int(s.end)
    end_after = int(s.end + 30)
    print(f"--- Transcript BEFORE GT [{start_before}s-{start_gt}s] ---")
    print(f"  {before_text[:200]}")
    print()
    print(f">>> TRANSCRIPT IN GT SEGMENT [{start_gt}s-{end_gt}s] <<<")
    print(f"  {gt_text}")
    print()
    print(f"--- Transcript AFTER GT [{end_gt}s-{end_after}s] ---")
    print(f"  {after_text[:200]}")
    print()
    print()
