"""Convert ReVA_V2 test_set.json to a flat list for Qwen inference.

Student task: complete every block marked TODO(student).
"""

import json
import os
from collections import Counter
from typing import Any, Iterator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REVA_ROOT = os.environ.get("REVA_ROOT", os.path.join(PROJECT_ROOT, "data", "reva_test"))
REVA_JSON = os.environ.get("REVA_JSON", os.path.join(REVA_ROOT, "test_set.json"))
OUTPUT_JSON = os.environ.get("REVA_PREPARED_JSON", os.path.join(os.path.dirname(__file__), "reva_v2_test_set.json"))


def get_video_duration(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception as exc:
        print(f"  Warning: could not read duration for {video_path}: {exc}")
        return 0.0


def _strip_dataset_prefix(file_path: str) -> str:
    """Remove optional #dataset/<name>/ prefixes from annotation paths."""
    path = file_path.strip()
    if path.startswith("#dataset/"):
        parts = path.split("/", 2)
        if len(parts) >= 3:
            path = parts[2]
    return path.lstrip("/")


def to_abs_path(file_path: str) -> str:
    """Convert an annotation video path to an absolute path under REVA_ROOT."""
    rel = _strip_dataset_prefix(file_path)
    if os.path.isabs(rel):
        return rel
    return os.path.join(REVA_ROOT, rel)


def to_rel_stem(file_path: str) -> str:
    """Convert an annotation video path to a relative stem without extension."""
    rel = _strip_dataset_prefix(file_path)
    stem, _ext = os.path.splitext(rel)
    return stem


def get_qa_id(qa: dict, video_item: dict, subcategory: str, question_idx: int) -> str:
    """Return a stable QA id for evaluation."""
    if qa.get("qa_id"):
        return str(qa["qa_id"])
    if qa.get("global_index") is not None:
        return f"REVA-G{int(qa['global_index']):06d}"
    stem = os.path.basename(to_rel_stem(video_item.get("file_path", "video")))
    abbr = (subcategory or "unk")[:3].upper()
    return f"REVA-{stem}-{abbr}-{question_idx:04d}"


def iter_flat_items(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield flat Qwen-evaluation items from nested ReVA test annotations."""
    for _video_key, video_item in data.get("videos", {}).items():
        file_path = video_item.get("file_path", "")
        abs_path = to_abs_path(file_path)
        if not os.path.isfile(abs_path):
            continue
        duration = get_video_duration(abs_path)
        video_id = to_rel_stem(file_path)
        mcq = video_item.get("mcq", {}) or {}
        for _category, question_types in mcq.items():
            if not isinstance(question_types, dict):
                continue
            for subcategory, qa_list in question_types.items():
                if not isinstance(qa_list, list):
                    continue
                for question_idx, qa in enumerate(qa_list):
                    if not isinstance(qa, dict):
                        continue
                    options = qa.get("options", {}) or {}
                    option_lines = "\n".join(
                        f"{label}. {options[label]}" for label in sorted(options.keys())
                    )
                    question_text = str(qa.get("question", "")).rstrip()
                    if option_lines:
                        question_text = f"{question_text}\n{option_lines}"
                    yield {
                        "id": get_qa_id(qa, video_item, subcategory, question_idx),
                        "video_id": video_id,
                        "duration": duration,
                        "question": question_text,
                        "answer": str(qa.get("correct_answer", "")).strip().upper(),
                        "question_type": subcategory,
                    }


def main():
    with open(REVA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    subcats = Counter()
    for item in iter_flat_items(data):
        items.append(item)
        subcats[item.get("question_type", "unknown")] += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(items)} QA items to {OUTPUT_JSON}")
    print("\nSubcategory distribution:")
    for subcat, count in sorted(subcats.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {subcat}: {count}")


if __name__ == "__main__":
    main()
