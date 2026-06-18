#!/usr/bin/env bash
set -euo pipefail

echo "=== Install system packages ==="
apt-get update -qq
apt-get install -y -qq ffmpeg unzip curl python3-pip >/dev/null 2>&1 || true

echo "=== Install Python packages ==="
python3 -m pip install --upgrade pip
python3 -m pip install --default-timeout=300 torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install --default-timeout=300 -r requirements.txt

echo "=== Download competition data ==="
: "${KAGGLE_BEARER_TOKEN:?Set KAGGLE_BEARER_TOKEN before downloading Kaggle data}"
mkdir -p /root/data/video-rag
cd /root/data
curl -L --max-time 1200 -o kaggle_data.zip \
  -H "Authorization: Bearer ${KAGGLE_BEARER_TOKEN}" \
  "https://www.kaggle.com/api/v1/competitions/data/download-all/multi-lingual-video-fragment-retrieval-challenge"
unzip -o -q kaggle_data.zip -d video-rag/
ls -la /root/data/video-rag/
