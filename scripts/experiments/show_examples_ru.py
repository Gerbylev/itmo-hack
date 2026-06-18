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

# Pick 5 diverse Russian examples from different topics and durations
np_seed = 123
samples = []

# Different topics
for topic in ["cooking-tutorials", "DIY", "travel-usa", "public-speaking", "first-aid"]:
    subset = train[train["topic"] == topic]
    if len(subset) > 0:
        samples.append(subset.sample(1, random_state=np_seed).iloc[0])

for i, s in enumerate(samples):
    vh = extract_hash(s.video_file)
    segs = th.get(vh, [])

    gt_segs = [seg for seg in segs if seg["end"] > s.start and seg["start"] < s.end]
    gt_text = " ".join(seg["text"].strip() for seg in gt_segs)
    if not gt_text:
        gt_text = "(ПУСТО - Whisper не распознал)"

    sep = "=" * 80
    print(sep)
    dur = int(s.duration)
    print(f"EXAMPLE {i+1} | Topic: {s.topic} | Duration: {dur}s")
    print(sep)
    print(f"Video: {s.video_file}")
    print(f"GT segment: {s.start}s - {s.end}s")
    print()
    print("ВОПРОС (RU):")
    print(f"  {s.question_ru}")
    print()
    print("ВОПРОС (EN):")
    print(f"  {s.question_en}")
    print()
    ans = s.answer_en[:400]
    print(f"ЭТАЛОННЫЙ ОТВЕТ (первые 400 символов):")
    print(f"  {ans}...")
    print()
    print(f"ТРАНСКРИПТ GT СЕГМЕНТА [{int(s.start)}s-{int(s.end)}s]:")
    print(f"  {gt_text}")
    print()
    print()
