from __future__ import annotations

import argparse

import numpy as np

from common import REAL_COUNTING_DIR, REAL_COUNTING_METADATA_PATH, REAL_MEASUREBENCH_DIR, REAL_MEASUREBENCH_METADATA_PATH, ensure_base_directories, save_image, write_csv, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch small real-data subsets for Measure-and-Render.")
    parser.add_argument("--counting-limit", type=int, default=50)
    parser.add_argument("--measure-limit", type=int, default=50)
    return parser.parse_args()


def infer_count(example: dict[str, object]) -> int | None:
    for key in ["count", "gt_count", "object_count"]:
        if key in example and example[key] is not None:
            return int(example[key])
    for key in ["points", "annotation_points", "dots"]:
        value = example.get(key)
        if isinstance(value, list):
            return len(value)
    annotations = example.get("annotations")
    if isinstance(annotations, list):
        return len(annotations)
    return None


def save_counting_subset(limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        return []
    from datasets import load_dataset

    dataset = load_dataset("isentropic/FSC147", split=f"default[:{limit}]")
    rows: list[dict[str, object]] = []
    skipped = 0
    for index, example in enumerate(dataset):
        image = example.get("image")
        count = infer_count(example)
        if image is None or count is None:
            skipped += 1
            continue
        sample_id = f"real_count_{index + 1:03d}"
        filename = f"{sample_id}.png"
        save_image(REAL_COUNTING_DIR / filename, np.array(image))
        rows.append(
            {
                "sample_id": sample_id,
                "filename": filename,
                "true_count": count,
                "overlap_level": "real",
                "notes": "source=isentropic/FSC147",
                "split": "real",
                "source_dataset": "FSC147",
            }
        )
    if skipped:
        print(f"Skipped {skipped} FSC147 items because no count annotation could be inferred.")
    return rows


def save_measurebench_subset(limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        return []
    from datasets import load_dataset

    dataset_dict = load_dataset("FlagEval/MeasureBench")
    split_name = "test" if "test" in dataset_dict else next(iter(dataset_dict.keys()))
    dataset = dataset_dict[split_name]
    rows: list[dict[str, object]] = []

    for example in dataset:
        if len(rows) >= limit:
            break
        image = example.get("image")
        if image is None:
            continue
        category = str(example.get("instrument_type", example.get("image_type", example.get("design", example.get("category", "unknown")))))
        question = str(example.get("question", example.get("prompt", "")))
        answer = example.get("answer", example.get("value", example.get("label", example.get("evaluator_kwargs", ""))))
        sample_id = f"measurebench_{len(rows) + 1:03d}"
        filename = f"{sample_id}.png"
        save_image(REAL_MEASUREBENCH_DIR / filename, np.array(image))
        rows.append(
            {
                "sample_id": sample_id,
                "filename": filename,
                "question": question,
                "answer": answer,
                "category": category,
                "notes": "source=FlagEval/MeasureBench",
                "split": "real",
                "source_dataset": "MeasureBench",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ensure_base_directories()
    notes: list[str] = []
    counting_rows: list[dict[str, object]] = []
    measure_rows: list[dict[str, object]] = []

    try:
        counting_rows = save_counting_subset(args.counting_limit)
    except Exception as exc:
        notes.append(f"Counting subset fetch failed: {exc}")
        print(f"Warning: counting subset fetch failed: {exc}")

    try:
        measure_rows = save_measurebench_subset(args.measure_limit)
    except Exception as exc:
        notes.append(f"MeasureBench subset fetch failed: {exc}")
        print(f"Warning: MeasureBench subset fetch failed: {exc}")

    if counting_rows:
        write_csv(
            REAL_COUNTING_METADATA_PATH,
            counting_rows,
            fieldnames=["sample_id", "filename", "true_count", "overlap_level", "notes", "split", "source_dataset"],
        )
    if measure_rows:
        write_csv(
            REAL_MEASUREBENCH_METADATA_PATH,
            measure_rows,
            fieldnames=["sample_id", "filename", "question", "answer", "category", "notes", "split", "source_dataset"],
        )

    sources_lines = [
        "Real counting subset source: https://huggingface.co/datasets/isentropic/FSC147",
        "Real measurement subset source: https://huggingface.co/datasets/FlagEval/MeasureBench",
        "The fetch script writes deterministic local copies plus metadata under data/real/ when downloads succeed.",
    ]
    if notes:
        sources_lines.extend(["", "Fetch notes:"])
        sources_lines.extend([f"- {note}" for note in notes])
    write_text(REAL_COUNTING_DIR.parent / "SOURCES.md", "\n".join(sources_lines) + "\n")
    print(f"Saved {len(counting_rows)} counting samples and {len(measure_rows)} MeasureBench samples.")


if __name__ == "__main__":
    main()
