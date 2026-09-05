#!/usr/bin/env bash
set -euo pipefail

PROJECT=<REPO>
PYTHON=python
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT}/additional_Qs/rq3_cf_rescue_20260830}"

mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${OUTPUT_DIR}/logs/linger_wrapper.log") 2>&1

export HF_HOME=<HF_CACHE>
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RQ3_TORCH_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

until nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; do
  echo "$(date --iso-8601=seconds) WAIT_GPU_DRIVER"
  sleep 30
done

echo "$(date --iso-8601=seconds) CF_RESCUE_QUEUE_START"
cd "${PROJECT}"
exec "${PYTHON}" -u additional_Qs/queue_rq3_cf_rescue.py \
  --output-dir "${OUTPUT_DIR}" --devices 0 1 2 3 --max-gpu-mib 24000 --num-shards 2
