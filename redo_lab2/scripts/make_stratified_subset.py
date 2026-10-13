#!/usr/bin/env python3
import argparse, copy, hashlib, json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reva_eval" / "data" / "rsvidqa"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/data/datasets/ReVA/test_set.json")
    ap.add_argument("--dataset-root", default="/data/datasets/ReVA")
    ap.add_argument("--output", default=str(ROOT / "redo_lab2/data/reva_stratified_110.json"))
    ap.add_argument("--manifest", default=str(ROOT / "redo_lab2/data/reva_stratified_110_manifest.json"))
    ap.add_argument("--seed", type=int, default=398)
    ap.add_argument("--per-type", type=int, default=10)
    args = ap.parse_args()

    os.environ["REVA_ROOT"] = args.dataset_root
    import prepare_reva_v2_test_set as prep

    with open(args.source, encoding="utf-8") as f:
        source = json.load(f)
    flat = list(prep.iter_flat_items(source))
    grouped = defaultdict(list)
    for item in flat:
        grouped[item["question_type"]].append(item)

    rng = random.Random(args.seed)
    selected = []
    for question_type in sorted(grouped):
        pool = grouped[question_type]
        if len(pool) < args.per_type:
            raise ValueError(f"{question_type}: only {len(pool)} items")
        selected.extend(rng.sample(pool, args.per_type))
    selected_ids = {item["id"] for item in selected}

    subset = copy.deepcopy(source)
    kept_ids = []
    videos = subset.get("videos", {})
    for video_key in list(videos):
        video_item = videos[video_key]
        mcq = video_item.get("mcq", {}) or {}
        for category in list(mcq):
            question_types = mcq[category]
            if not isinstance(question_types, dict):
                del mcq[category]
                continue
            for subcategory in list(question_types):
                qa_list = question_types[subcategory]
                if not isinstance(qa_list, list):
                    del question_types[subcategory]
                    continue
                filtered = []
                for idx, qa in enumerate(qa_list):
                    qa_id = prep.get_qa_id(qa, video_item, subcategory, idx)
                    if qa_id in selected_ids:
                        filtered.append(qa)
                        kept_ids.append(qa_id)
                if filtered:
                    question_types[subcategory] = filtered
                else:
                    del question_types[subcategory]
            if not question_types:
                del mcq[category]
        if not mcq:
            del videos[video_key]

    if set(kept_ids) != selected_ids or len(kept_ids) != len(selected_ids):
        raise RuntimeError("Could not reconstruct selected IDs in nested annotation")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selected_sorted = sorted(selected, key=lambda x: x["id"])
    counts = Counter(x["question_type"] for x in selected)
    missing_videos = [x["video_id"] for x in selected if not (Path(args.dataset_root) / f"{x['video_id']}.mp4").is_file()]
    manifest = {
        "source": args.source,
        "dataset_root": args.dataset_root,
        "selection": "independent uniform sample without replacement within each question_type",
        "seed": args.seed,
        "questions_per_type": args.per_type,
        "num_questions": len(selected),
        "num_videos": len(videos),
        "counts_by_question_type": dict(sorted(counts.items())),
        "selected_ids": [x["id"] for x in selected_sorted],
        "selected_ids_sha256": hashlib.sha256("\n".join(x["id"] for x in selected_sorted).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "missing_video_count": len(missing_videos),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
