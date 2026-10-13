#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BIN=${CONDA_BIN:-/opt/tljh/user/bin/conda}
CONDA_ENV=${CONDA_ENV:-qwen2}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH}
MODEL_BASE=${MODEL_BASE:-}
EVAL_NAME=${EVAL_NAME:?Set EVAL_NAME}
GPU_IDS=${GPU_IDS:-1,3}
MAX_FRAMES=${MAX_FRAMES:-4}
MAX_PIXELS=${MAX_PIXELS:-50176}
BACKEND=${BACKEND:-transformers}
REVA_ROOT=${REVA_ROOT:-/data/datasets/ReVA}
REVA_JSON=${REVA_JSON:-$ROOT/redo_lab2/data/reva_stratified_110.json}
GT_FILE=${GT_FILE:-$ROOT/redo_lab2/data/reva_stratified_110_flat.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/redo_lab2/outputs}
IFS=, read -ra GPUS <<< "$GPU_IDS"
NUM_CHUNKS=${#GPUS[@]}
OUT="$OUTPUT_ROOT/$EVAL_NAME/$EVAL_NAME"
LOG="$ROOT/redo_lab2/logs/$EVAL_NAME.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
RESUME=${RESUME:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
export EVAL_BATCH_SIZE
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if compgen -G "$OUT/*.json" >/dev/null && [ "$RESUME" != "1" ]; then
  echo "Refusing to overwrite existing JSON in $OUT" >&2
  echo "Set RESUME=1 to continue from saved shard predictions." >&2
  exit 2
fi
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
echo "started_utc=$(date -u +%FT%TZ)"
echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "model_path=$MODEL_PATH"
echo "model_base=$MODEL_BASE"
echo "reva_json=$REVA_JSON"
echo "gt_file=$GT_FILE"
echo "gpu_ids=$GPU_IDS"
echo "num_chunks=$NUM_CHUNKS"
echo "max_frames=$MAX_FRAMES"
echo "max_pixels=$MAX_PIXELS"
echo "backend=$BACKEND"
echo "resume=$RESUME"
echo "eval_batch_size=$EVAL_BATCH_SIZE"
sha256sum "$REVA_JSON" "$GT_FILE"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

cd "$ROOT/reva_eval"
REVA_ROOT="$REVA_ROOT" REVA_JSON="$REVA_JSON" REVA_PREPARED_JSON="$GT_FILE" \
  "$CONDA_BIN" run -n "$CONDA_ENV" python3 data/rsvidqa/prepare_reva_v2_test_set.py
export MY_PROMPT_TEMPLATE="This is a video with duration {duration} seconds.
Please carefully watch the video and answer the multiple-choice question below.
Think step-by-step within <think> </think> tags, then provide only the letter of the correct option within <answer> </answer> tags.
Question: {input_text}"
{
  echo "MODEL_PATH=$MODEL_PATH"; echo "MODEL_BASE=$MODEL_BASE"; echo "REVA_ROOT=$REVA_ROOT"
  echo "REVA_JSON=$REVA_JSON"; echo "GT_FILE=$GT_FILE"; echo "GPU_IDS=$GPU_IDS"
  echo "NUM_CHUNKS=$NUM_CHUNKS"; echo "MAX_FRAMES=$MAX_FRAMES"; echo "MAX_PIXELS=$MAX_PIXELS"
  echo "BACKEND=$BACKEND"; echo "RESUME=$RESUME"; echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
} > "$OUT/run_config.txt"

pids=()
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}
  extra=(); [ -n "$MODEL_BASE" ] && extra+=(--model-base "$MODEL_BASE")
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$CONDA_ENV" python3 inference_vllm_origin_number.py \
    --model-path "$MODEL_PATH" "${extra[@]}" --video_dir "$REVA_ROOT" --gt_file "$GT_FILE" \
    --output_dir "$OUT" --output_name pred --num-chunks "$NUM_CHUNKS" --chunk-idx "$idx" \
    --max_pixels "$MAX_PIXELS" --max_frames "$MAX_FRAMES" --backend "$BACKEND" &
  pids+=("$!")
  echo "launched chunk=$idx physical_gpu=$gpu pid=$!"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "Inference failed"; exit 1; }
"$CONDA_BIN" run -n "$CONDA_ENV" python3 "$ROOT/scripts/score_reva_predictions.py" --output-dir "$OUT" --gt-file "$GT_FILE"
END=$(date +%s)
echo "finished_utc=$(date -u +%FT%TZ)"
echo "elapsed_seconds=$((END-START))"
cat "$OUT/result.csv"
