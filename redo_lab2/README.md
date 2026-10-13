# CSE398 Lab 2 Redo — ReVA VQA

This package contains the Lab 2 rerun completed on MAGIC on October 12–13, 2026.
The scored comparison uses the **full 4,000-question ReVA test set**.

## Headline results (4,000 questions, same IDs)

- Qwen3-VL-4B-Instruct base: 2702/4000 (67.55%)
- Explicitly retrained Qwen3-VL LoRA: 2701/4000 (67.53%)
- VILA1.5-3B: 1942/4000 (48.55%)
- Parseable answers: base 3984/4000, LoRA 3977/4000, VILA 4000/4000
  (missing/unparsed answers counted as wrong)

All three models were scored on the same 4,000 question IDs. A 110-question
balanced subset (seed 398) was used earlier as a smoke/debug run; those subset
numbers are **not** the reported result.

## Package contents

- `report/CSE398_Lab2_ReVA_Redo_Report.pdf` — submission-ready report
- `report/model_comparison.csv` — overall model comparison with Wilson 95% CIs
- `report/verification.json` — ID matching, McNemar tests, and per-type counts
- `scripts/` — full-4000 pipeline, two-GPU/3-GPU Qwen runner, VILA Turing patch
- `outputs/` — scored CSV summaries
- `training_metadata/` — saved LoRA adapter configuration and trainer state

The full ReVA videos, base-model weights, LoRA tensors, and raw 4,000-question
prediction shards are excluded because of size. Their paths/model IDs and all
experimental settings are recorded in the report.
