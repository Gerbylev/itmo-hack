# ITMO Hack: multilingual video fragment retrieval

Code for the Multi-lingual Video Fragment Retrieval Challenge.

The task is to find relevant time fragments in videos using transcript text and natural-language questions. The solution uses transcript chunking, SentenceTransformers embeddings, FAISS search, and timestamp post-processing.

## What to look at

1. `kaggle/solution_e5ft_s95.py` - final compact Kaggle solution.
2. `scripts/train_embeddings.py` - fine-tuning pipeline for embedding models.
3. `scripts/run_baseline.py` - baseline retrieval pipeline.
4. `tools/setup_and_run.sh` - dependency and data setup helper.

Everything else from the research phase was removed from the current tree to keep the repository readable. It is still recoverable from git history, especially commit `5b7de4d`.

## Structure

```text
.
├── kaggle/
│   └── solution_e5ft_s95.py
├── scripts/
│   ├── run_baseline.py
│   └── train_embeddings.py
├── tools/
│   └── setup_and_run.sh
├── README.md
└── requirements.txt
```

## Data layout

Most scripts expect the competition data at:

```text
/root/data/video-rag
```

The Kaggle solution expects:

```text
/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag
```

Expected files:

- `transcripts.pkl`
- `video_files.csv`
- `train/train_qa.csv`
- `test/test.csv`

## Main models

- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- fine-tuned model: `olegGerbylev/e5-large-video-retrieval-ft-v2`

## Result note

The final compact solution in `kaggle/solution_e5ft_s95.py` records:

- Train `FinalScore`: `0.5625`
- Kaggle score: `0.471`
