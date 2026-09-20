#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    tagged = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = tagged.group(1).strip() if tagged else text

    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*([A-Ha-h])\b",
        r"^\s*([A-Ha-h])\s*$",
        r"\b([A-Ha-h])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, candidate, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    preds: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(output_dir).glob("*.json")):
        if path.name == "result.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = item.get("id")
                if item_id is None:
                    continue
                preds[str(item_id)] = item
    return preds


def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    results: dict[str, dict[str, Any]] = {}
    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)

    completed = 0
    correct = 0
    total = len(gt_list)

    for gt in gt_list:
        gt_id = str(gt["id"])
        qtype = str(gt.get("question_type", "unknown"))
        gt_answer = str(gt.get("answer", "")).strip().upper()
        by_type_total[qtype] += 1

        pred_item = preds.get(gt_id)
        pred_text = ""
        if pred_item is not None:
            pred_text = str(pred_item.get("pred", pred_item.get("prediction", "")))

        pred_letter = extract_letter(pred_text) if pred_text else ""
        if pred_letter:
            completed += 1

        acc = int(bool(pred_letter) and pred_letter == gt_answer)
        if acc:
            correct += 1
            by_type_correct[qtype] += 1

        results[gt_id] = {
            "gt": gt_answer,
            "pred_letter": pred_letter,
            "pred": pred_text,
            "acc": acc,
            "question_type": qtype,
        }

    def pct(num: int, den: int) -> str:
        value = (100.0 * num / den) if den else 0.0
        return f"{num}/{den} = {value:.2f}%"

    lines = [
        f"Completed: {pct(completed, total)}",
        f"Total: {pct(correct, total)}",
    ]
    for qtype in sorted(by_type_total.keys()):
        lines.append(f"{qtype}: {pct(by_type_correct[qtype], by_type_total[qtype])}")

    csv_text = "\n".join(lines) + "\n"
    return results, csv_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
