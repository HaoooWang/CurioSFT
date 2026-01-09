#!/usr/bin/env bash
set -euo pipefail

HF_DATA_REPO="Hao0oWang/CurioSFT_Data"
HF_MODEL_REPO="Hao0oWang/Qwen2.5-Math-7B-16k-think"

DATA_DIR_REL="data"
MODELS_DIR_REL="models"

# Move to repo root so relative paths resolve consistently.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Dependency check: requires huggingface-cli.
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[error] huggingface-cli not found. Install: pip install -U huggingface_hub"
  exit 1
fi

# Step 1: download dataset into data/.
echo "==> Step 1: download dataset into '${DATA_DIR_REL}'"
mkdir -p "${DATA_DIR_REL}"
if [ -z "${HF_DATA_REPO}" ]; then
  echo "[error] HF_DATA_REPO is empty"
  exit 1
fi


huggingface-cli download "${HF_DATA_REPO}"\
  --repo-type dataset \
  --local-dir "${DATA_DIR_REL}"

echo "[done] Dataset downloaded to '${DATA_DIR_REL}'"
echo "Preview:"
ls -lh "${DATA_DIR_REL}" || true

# Step 2: download model into models/.
echo "==> Step 2: download model into '${MODELS_DIR_REL}'"
mkdir -p "${MODELS_DIR_REL}"
if [ -z "${HF_MODEL_REPO}" ]; then
  echo "[error] HF_MODEL_REPO is empty"
  exit 1
fi

# Convert repo id to a filesystem-friendly directory name.
MODEL_DEST="${MODELS_DIR_REL}/"

huggingface-cli download "${HF_MODEL_REPO}" \
  --local-dir "${MODEL_DEST}"

echo "[done] Model downloaded to '${MODEL_DEST}'"
echo "Preview:"
ls -lh "${MODEL_DEST}" || true

echo "All done"
