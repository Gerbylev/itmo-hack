# ITMO Hack: multilingual video fragment retrieval

This repository contains experiments and final inference code for the Multi-lingual Video Fragment Retrieval Challenge.

The task is to retrieve relevant video fragments for natural-language questions using video transcripts. The main approach is dense retrieval over transcript chunks with SentenceTransformers embeddings, FAISS search, optional reranking, and timestamp post-processing.

## What is included

- `finetune_and_eval.py` - fine-tuning pipeline for `BAAI/bge-m3` and `intfloat/multilingual-e5-large` on transcript/question pairs.
- `run_baseline.py` - baseline dense retrieval pipeline.
- `dense_chunks.py`, `grid_search.py`, `experiments*.py` - chunking and retrieval experiments.
- `run_nb.py`, `run_submission_nb.py`, `radical_sr.py` - stronger inference pipelines with reranking and timestamp refinement.
- `finetune_results/solution_e5ft_s95.py` - compact final Kaggle solution using `olegGerbylev/e5-large-video-retrieval-ft-v2`.
- `finetune_results/kaggle_submission_notebook.py` - Kaggle-oriented submission script.

Generated submissions, local datasets, model weights, virtual environments, IDE metadata, and credentials are intentionally excluded from the repository.

## Data layout

Most scripts expect the competition data at one of these paths:

```text
/root/data/video-rag
/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag
```

Expected files include:

- `transcripts.pkl`
- `video_files.csv`
- `train/train_qa.csv`
- `test/test.csv`

## Main models

- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- `BAAI/bge-reranker-v2-m3`
- `cross-encoder/ms-marco-MiniLM-L-12-v2`
- fine-tuned model: `olegGerbylev/e5-large-video-retrieval-ft-v2`

## Result note

The final compact solution in `finetune_results/solution_e5ft_s95.py` records:

- Train `FinalScore`: `0.5625`
- Kaggle score: `0.471`
