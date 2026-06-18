# ITMO Hack: multilingual video fragment retrieval

Experiments and final inference code for the Multi-lingual Video Fragment Retrieval Challenge.

The task is to retrieve relevant video fragments for natural-language questions using video transcripts. The main approach is dense retrieval over transcript chunks with SentenceTransformers embeddings, FAISS search, optional reranking, and timestamp post-processing.

## Repository structure

```text
.
├── kaggle/                 # compact Kaggle submission scripts
├── notebooks/              # cleaned exploratory notebooks
├── scripts/
│   ├── experiments/        # metric checks, grid search, manual analysis
│   ├── inference/          # retrieval, reranking, timestamp refinement
│   └── training/           # embedding fine-tuning and evaluation
├── tools/                  # setup helpers
├── README.md
└── requirements.txt
```

## Main entry points

- `kaggle/solution_e5ft_s95.py` - compact final solution using `olegGerbylev/e5-large-video-retrieval-ft-v2`.
- `scripts/training/finetune_and_eval.py` - fine-tuning pipeline for `BAAI/bge-m3` and `intfloat/multilingual-e5-large`.
- `scripts/inference/run_baseline.py` - baseline dense retrieval pipeline.
- `scripts/inference/run_nb.py` and `scripts/inference/run_submission_nb.py` - stronger inference pipelines with reranking and timestamp refinement.
- `scripts/inference/radical_sr.py` - retrieval pipeline with cross-encoder reranking experiments, including MiniLM.
- `tools/setup_and_run.sh` - dependency and competition data setup helper. Requires `KAGGLE_BEARER_TOKEN` in the environment.

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

The final compact solution in `kaggle/solution_e5ft_s95.py` records:

- Train `FinalScore`: `0.5625`
- Kaggle score: `0.471`
