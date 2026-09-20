import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()


def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""
    with open(question_file, encoding="utf-8") as f:
        data = json.load(f)

    instances: list[dict[str, Any]] = []
    for video_key, video_item in data.get("videos", {}).items():
        file_path = video_item.get("file_path", "")
        dataset_name = video_item.get("dataset_name")
        stem = Path(file_path).stem
        id_stem = dataset_name or stem or str(video_key)
        mcq = video_item.get("mcq", {}) or {}
        for category, question_types in mcq.items():
            if not isinstance(question_types, dict):
                continue
            for subcategory, qa_list in question_types.items():
                if not isinstance(qa_list, list):
                    continue
                for question_idx, qa in enumerate(qa_list):
                    if not isinstance(qa, dict):
                        continue
                    if qa.get("qa_id"):
                        qa_id = str(qa["qa_id"])
                    elif qa.get("global_index") is not None:
                        qa_id = f"REVA-G{int(qa['global_index']):06d}"
                    else:
                        abbr = (subcategory or "unk")[:3].upper()
                        qa_id = f"REVA-{id_stem}-{abbr}-{question_idx:04d}"

                    instances.append(
                        {
                            "qa_id": qa_id,
                            "video_id": video_key,
                            "video_path": file_path,
                            "dataset_name": dataset_name,
                            "category": category,
                            "subcategory": subcategory,
                            "question": qa.get("question", ""),
                            "options": qa.get("options", {}) or {},
                            "correct_answer": str(qa.get("correct_answer", "")).strip().upper(),
                            "reasoning": qa.get("reasoning", ""),
                            "example": qa.get("example", ""),
                        }
                    )
    return instances


def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""
    candidates = [raw_path, os.path.join(dataset_root, raw_path.lstrip("/"))]
    if raw_path.startswith(dataset_prefix):
        rest = raw_path[len(dataset_prefix) :].lstrip("/")
        candidates.append(os.path.join(dataset_root, rest))
        parts = rest.split("/", 1)
        if len(parts) == 2:
            candidates.append(os.path.join(dataset_root, parts[1]))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return candidates[-1] if candidates else raw_path


def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""
    lines = [question]
    for label in sorted(options.keys()):
        lines.append(f"{label}. {options[label]}")
    lines.append("Answer with only the option letter from the given choices.")
    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str]) -> str | None:
    """Parse a choice letter from VILA raw response."""
    if not response:
        return None

    valid = {k.upper() for k in options.keys()}

    answer_match = re.search(
        r"(?:final\s+)?answer(?:\s+is)?\s*[:=]?\s*([A-Da-d])\b",
        response,
        flags=re.IGNORECASE,
    )
    if answer_match:
        letter = answer_match.group(1).upper()
        if letter in valid:
            return letter

    lone = re.fullmatch(r"\s*([A-Da-d])\s*[.)]?\s*", response, flags=re.IGNORECASE)
    if lone:
        letter = lone.group(1).upper()
        if letter in valid:
            return letter

    response_lower = response.lower()
    ranked = sorted(options.items(), key=lambda kv: len(str(kv[1])), reverse=True)
    for label, text in ranked:
        if str(text).lower() and str(text).lower() in response_lower:
            return str(label).upper()
    return None


def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""
    num_questions = len(records)
    num_answered = sum(1 for r in records if r.get("pred_letter") is not None)
    num_correct = sum(1 for r in records if r.get("is_correct"))
    accuracy = (num_correct / num_questions) if num_questions else 0.0

    def group_stats(key: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(str(record.get(key, "unknown")), []).append(record)
        out: dict[str, dict[str, float | int]] = {}
        for name, items in groups.items():
            n = len(items)
            c = sum(1 for item in items if item.get("is_correct"))
            out[name] = {
                "num_questions": n,
                "num_correct": c,
                "accuracy": (c / n) if n else 0.0,
            }
        return out

    return {
        "num_questions": num_questions,
        "num_answered": num_answered,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "by_category": group_stats("category"),
        "by_subcategory": group_stats("subcategory"),
    }


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
