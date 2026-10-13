#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH=/opt/tljh/user/bin:$PATH

STATUS="$ROOT/redo_lab2/logs/full_4000_status.txt"
PIPELOG="$ROOT/redo_lab2/logs/full_4000_pipeline.log"
FULL_JSON=/data/datasets/ReVA/test_set.json
FULL_FLAT="$ROOT/redo_lab2/data/reva_full_4000_flat.json"

update_status() {
  printf "%s phase=%s\n" "$(date -u +%FT%TZ)" "$1" | tee "$STATUS"
}

on_error() {
  code=$?
  update_status "FAILED_exit_$code"
  exit "$code"
}
trap on_error ERR

exec > >(tee -a "$PIPELOG") 2>&1

echo "=== Full ReVA 4000-question pipeline ==="
echo "started_utc=$(date -u +%FT%TZ)"
echo "git_commit=$(git rev-parse HEAD)"
echo "full_flat_sha256=$(sha256sum "$FULL_FLAT" | cut -d' ' -f1)"
echo "qwen_gpus=1,2,3; vila_gpu=0_if_free_else_wait; max_frames=4; max_pixels=50176; eval_batch_size=1"

python3 - <<'PY'
import json
p = "redo_lab2/data/reva_full_4000_flat.json"
d = json.load(open(p))
assert len(d) == 4000 and len({x["id"] for x in d}) == 4000
print("verified_questions=4000 verified_unique_ids=4000")
PY

update_status QWEN_BASE
RESUME=1 \
EVAL_BATCH_SIZE=1 \
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
EVAL_NAME=full_qwen_base_n4000 \
GPU_IDS=1,2,3 \
MAX_FRAMES=4 \
MAX_PIXELS=50176 \
BACKEND=transformers \
REVA_ROOT=/data/datasets/ReVA \
REVA_JSON="$FULL_JSON" \
GT_FILE="$FULL_FLAT" \
bash redo_lab2/scripts/run_qwen_eval_2gpu.sh

update_status QWEN_LORA
RESUME=0 \
EVAL_BATCH_SIZE=1 \
MODEL_PATH="$ROOT/redo_lab2/outputs/qwen_reva_sft_explicit" \
MODEL_BASE=Qwen/Qwen3-VL-4B-Instruct \
EVAL_NAME=full_qwen_lora_n4000 \
GPU_IDS=1,2,3 \
MAX_FRAMES=4 \
MAX_PIXELS=50176 \
BACKEND=transformers \
REVA_ROOT=/data/datasets/ReVA \
REVA_JSON="$FULL_JSON" \
GT_FILE="$FULL_FLAT" \
bash redo_lab2/scripts/run_qwen_eval_2gpu.sh

update_status VILA
rm -rf "$ROOT/redo_lab2/outputs/full_vila_n4000"
CONDA_ENV=vila \
VILA_REPO="$HOME/VILA" \
MODEL_PATH=Efficient-Large-Model/VILA1.5-3b \
REVA_ROOT=/data/datasets/ReVA \
REVA_JSON="$FULL_JSON" \
OUTPUT_DIR="$ROOT/redo_lab2/outputs/full_vila_n4000" \
NUM_VIDEO_FRAMES=4 \
GPU=1 \
bash scripts/run_eval_reva_vila.sh

update_status VERIFY
python3 - <<'PY'
import json
root = "redo_lab2/outputs"
b = json.load(open(f"{root}/full_qwen_base_n4000/full_qwen_base_n4000/result.json"))
s = json.load(open(f"{root}/full_qwen_lora_n4000/full_qwen_lora_n4000/result.json"))
v = [
    json.loads(x)
    for x in open(f"{root}/full_vila_n4000/outputs.jsonl")
    if x.strip()
]
assert len(b) == len(s) == len(v) == 4000
assert set(b) == set(s) == {x["qa_id"] for x in v}
print("verified_all_models=4000 identical_ids=true")
print("base_correct", sum(x["acc"] for x in b.values()))
print("lora_correct", sum(x["acc"] for x in s.values()))
print("vila_correct", sum(x["is_correct"] for x in v))
PY

update_status COMPLETE
echo "finished_utc=$(date -u +%FT%TZ)"
