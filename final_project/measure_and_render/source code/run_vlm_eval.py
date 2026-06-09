from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from angle_task import run_angle_benchmark
from common import BASE_CONDITIONS, VLM_OUTPUT_DIR, ensure_base_directories, task_output_paths, write_csv, write_text
from counting_task import run_counting_benchmark
from harness import load_manifest, summarize_numeric_predictions
from vlm_adapters import build_adapter


LONG_FIELDNAMES = [
    "task",
    "sample_id",
    "split",
    "source_dataset",
    "difficulty",
    "notes",
    "model_name",
    "model_id",
    "condition",
    "base_condition",
    "perturbation",
    "image_path",
    "prompt",
    "expected_value",
    "tolerance",
    "measurement_text",
    "measurement_value",
    "predicted_text",
    "predicted_value",
    "parse_ok",
    "error",
    "abs_error",
    "exact",
    "within_tolerance",
    "follow_injected",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-VLM evaluation on Measure-and-Render manifests.")
    parser.add_argument("--task", choices=["counting", "angle", "all"], default="all")
    parser.add_argument("--model", required=True, help="Model alias or adapter prefix, for example qwen2.5-vl-3b or openai:gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--conditions", default="all", help="Comma-separated base conditions. Default: all")
    parser.add_argument("--split", choices=["all", "synthetic", "real"], default="all")
    parser.add_argument("--perturbations", choices=["all", "zero", "nonzero"], default="all")
    parser.add_argument("--refresh-manifests", action="store_true")
    return parser.parse_args()


def ensure_manifests(task_name: str, refresh: bool) -> Path:
    manifest_path = task_output_paths(task_name)["manifest_path"]
    if refresh or not manifest_path.exists():
        if task_name == "counting":
            run_counting_benchmark()
        else:
            run_angle_benchmark()
    return manifest_path


def load_variants(task_name: str, refresh: bool) -> list[dict[str, object]]:
    return load_manifest(ensure_manifests(task_name, refresh))


def filter_variants(
    variants: list[dict[str, object]],
    base_conditions: set[str],
    split: str,
    perturbations: str,
    limit: int | None,
    seed: int,
) -> list[dict[str, object]]:
    filtered = [
        variant
        for variant in variants
        if variant["base_condition"] in base_conditions
        and (split == "all" or variant["split"] == split)
        and (
            perturbations == "all"
            or (perturbations == "zero" and abs(float(variant["perturbation"])) < 1e-9)
            or (perturbations == "nonzero" and abs(float(variant["perturbation"])) >= 1e-9)
        )
    ]
    filtered = sorted(filtered, key=lambda variant: (variant["task"], variant["split"], variant["sample_id"], variant["base_condition"], float(variant["perturbation"])))
    if limit is not None:
        rng = np.random.default_rng(seed)
        sample_keys = sorted({(str(variant["task"]), str(variant["split"]), str(variant["sample_id"])) for variant in filtered})
        if len(sample_keys) > limit:
            chosen = {sample_keys[index] for index in sorted(rng.choice(len(sample_keys), size=limit, replace=False).tolist())}
            filtered = [variant for variant in filtered if (str(variant["task"]), str(variant["split"]), str(variant["sample_id"])) in chosen]
    return filtered


def load_existing_long(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def result_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["task"]),
        str(row["sample_id"]),
        str(row["condition"]),
        str(row["split"]),
    )


def merge_results(existing: list[dict[str, str]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in existing:
        merged[result_key(row)] = dict(row)
    for row in new_rows:
        merged[result_key(row)] = row
    return [merged[key] for key in sorted(merged.keys())]


def evaluate_variants(variants: list[dict[str, object]], model_name: str) -> list[dict[str, object]]:
    adapter = build_adapter(model_name)
    rows: list[dict[str, object]] = []
    for index, variant in enumerate(variants, start=1):
        image_path = Path(str(variant["image_path"]))
        prompt = str(variant["prompt"])
        reply = adapter.answer(image_path, prompt)
        parsed_value = adapter.parse_number(reply)
        expected_value = float(variant["expected_value"])
        tolerance = float(variant["tolerance"])
        perturbation = float(variant["perturbation"])
        measurement_value = None if variant["measurement_value"] is None else float(variant["measurement_value"])

        error = None if parsed_value is None else parsed_value - expected_value
        abs_error = None if error is None else abs(error)
        exact = 0.0 if abs_error is None else float(abs_error <= 1e-6)
        within_tolerance = 0.0 if abs_error is None else float(abs_error <= tolerance)

        follow_tolerance = 0.5 if str(variant["unit"]) == "degrees" else 1e-6
        follow_injected = 0.0
        if parsed_value is not None and measurement_value is not None and abs(perturbation) > 1e-9:
            follow_injected = float(abs(parsed_value - measurement_value) <= follow_tolerance)

        row = {
            "task": str(variant["task"]),
            "sample_id": str(variant["sample_id"]),
            "split": str(variant["split"]),
            "source_dataset": str(variant["source_dataset"]),
            "difficulty": str(variant["difficulty"]),
            "notes": str(variant["notes"]),
            "model_name": model_name,
            "model_id": adapter.model_id,
            "condition": str(variant["condition"]),
            "base_condition": str(variant["base_condition"]),
            "perturbation": f"{perturbation:.1f}",
            "image_path": str(image_path),
            "prompt": prompt,
            "expected_value": f"{expected_value:.4f}",
            "tolerance": f"{tolerance:.4f}",
            "measurement_text": str(variant["measurement_text"]),
            "measurement_value": "" if measurement_value is None else f"{measurement_value:.4f}",
            "predicted_text": reply,
            "predicted_value": "" if parsed_value is None else f"{parsed_value:.4f}",
            "parse_ok": "1" if parsed_value is not None else "0",
            "error": "" if error is None else f"{error:.4f}",
            "abs_error": "" if abs_error is None else f"{abs_error:.4f}",
            "exact": f"{exact:.4f}",
            "within_tolerance": f"{within_tolerance:.4f}",
            "follow_injected": f"{follow_injected:.4f}",
        }
        rows.append(row)
        print(f"[{index}/{len(variants)}] {row['task']} {row['sample_id']} {row['condition']} -> {row['predicted_text']}")
    return rows


def summarize_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["task"]),
                str(row["split"]),
                str(row["model_name"]),
                str(row["model_id"]),
                str(row["condition"]),
                str(row["base_condition"]),
            )
        ].append(row)

    summary_rows: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        task, split, model_name, model_id, condition, base_condition = key
        parsed_pairs: list[tuple[float, float]] = []
        follow_values: list[float] = []
        measurement_values: list[float] = []
        perturbation = float(group_rows[0]["perturbation"])
        source_datasets = sorted({str(row["source_dataset"]) for row in group_rows})
        for row in group_rows:
            predicted_value = row["predicted_value"]
            if predicted_value != "":
                parsed_pairs.append((float(predicted_value), float(row["expected_value"])))
            if float(row["follow_injected"]) > 0.0 or abs(float(row["perturbation"])) > 1e-9:
                follow_values.append(float(row["follow_injected"]))
            if row["measurement_value"] != "":
                measurement_values.append(float(row["measurement_value"]))
        tolerance = float(group_rows[0]["tolerance"])
        metrics = summarize_numeric_predictions(parsed_pairs, exact_tolerance=tolerance)
        parse_rate = 0.0 if not group_rows else sum(int(row["parse_ok"]) for row in group_rows) / len(group_rows)
        follow_rate = float(np.mean(follow_values)) if follow_values else 0.0
        summary_rows.append(
            {
                "task": task,
                "split": split,
                "source_dataset": ";".join(source_datasets),
                "model_name": model_name,
                "model_id": model_id,
                "condition": condition,
                "base_condition": base_condition,
                "perturbation": f"{perturbation:.1f}",
                "sample_count": metrics.sample_count,
                "parse_rate": f"{parse_rate:.4f}",
                "exact_accuracy": f"{metrics.exact_accuracy:.4f}",
                "tolerance_accuracy": f"{metrics.tolerance_accuracy:.4f}",
                "mae": f"{metrics.mae:.4f}",
                "rmse": f"{metrics.rmse:.4f}",
                "bias": f"{metrics.bias:.4f}",
                "mae_ci_low": f"{metrics.mae_ci_low:.4f}",
                "mae_ci_high": f"{metrics.mae_ci_high:.4f}",
                "follow_rate": f"{follow_rate:.4f}",
            }
        )
    return summary_rows


def save_condition_bar_chart(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    filtered = [row for row in summary_rows if float(row["perturbation"]) == 0.0 and row["split"] == "synthetic"]
    if not filtered:
        return
    tasks = sorted({row["task"] for row in filtered})
    for task in tasks:
        task_rows = [row for row in filtered if row["task"] == task]
        model_names = sorted({row["model_name"] for row in task_rows})
        conditions = [condition for condition in BASE_CONDITIONS if any(row["base_condition"] == condition for row in task_rows)]
        fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 4 * len(model_names)), sharex=True)
        if len(model_names) == 1:
            axes = [axes]
        for axis, model_name in zip(axes, model_names):
            model_rows = [row for row in task_rows if row["model_name"] == model_name]
            values = []
            for condition in conditions:
                match = next((row for row in model_rows if row["base_condition"] == condition and float(row["perturbation"]) == 0.0), None)
                values.append(0.0 if match is None else float(match["tolerance_accuracy"]))
            axis.bar(np.arange(len(conditions)), values, color="#4c72b0")
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Accuracy")
            axis.set_title(f"{task}: {model_name}")
            axis.grid(True, axis="y", alpha=0.25)
        axes[-1].set_xticks(np.arange(len(conditions)))
        axes[-1].set_xticklabels(conditions, rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_condition_accuracy.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_raw_difficulty_figure(long_rows: list[dict[str, object]], output_dir: Path) -> None:
    filtered = [row for row in long_rows if row["base_condition"] == "raw" and row["predicted_value"] != "" and row["split"] == "synthetic"]
    if not filtered:
        return
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in filtered:
        grouped[(row["task"], row["model_name"], row["difficulty"])].append(float(row["within_tolerance"]))

    task_names = sorted({key[0] for key in grouped})
    for task_name in task_names:
        task_keys = [key for key in grouped if key[0] == task_name]
        model_names = sorted({key[1] for key in task_keys})
        difficulty_order = sorted({key[2] for key in task_keys})
        fig, ax = plt.subplots(figsize=(11, 5))
        for model_name in model_names:
            values = [float(np.mean(grouped.get((task_name, model_name, difficulty), [0.0]))) for difficulty in difficulty_order]
            ax.plot(np.arange(len(difficulty_order)), values, marker="o", linewidth=2, label=model_name)
        ax.set_xticks(np.arange(len(difficulty_order)))
        ax.set_xticklabels(difficulty_order)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Raw-condition accuracy")
        ax.set_title(f"{task_name}: raw VLM accuracy by synthetic difficulty")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{task_name}_raw_by_difficulty.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_base_directories()
    VLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = ["counting", "angle"] if args.task == "all" else [args.task]
    base_conditions = set(BASE_CONDITIONS if args.conditions == "all" else [item.strip() for item in args.conditions.split(",") if item.strip()])

    variants: list[dict[str, object]] = []
    for task_name in tasks:
        variants.extend(load_variants(task_name, refresh=args.refresh_manifests))
    filtered_variants = filter_variants(variants, base_conditions, args.split, args.perturbations, args.limit, args.seed)

    new_rows = evaluate_variants(filtered_variants, args.model)

    results_long_path = VLM_OUTPUT_DIR / "results_long.csv"
    existing_rows = load_existing_long(results_long_path)
    merged_rows = merge_results(existing_rows, new_rows)
    write_csv(results_long_path, merged_rows, fieldnames=LONG_FIELDNAMES)

    summary_rows = summarize_results(merged_rows)
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    if summary_fieldnames:
        write_csv(VLM_OUTPUT_DIR / "summary.csv", summary_rows, fieldnames=summary_fieldnames)
    save_condition_bar_chart(summary_rows, VLM_OUTPUT_DIR)
    save_raw_difficulty_figure(merged_rows, VLM_OUTPUT_DIR)

    write_text(
        VLM_OUTPUT_DIR / "README.txt",
        "results_long.csv stores one row per sample x condition x model. summary.csv stores grouped VLM metrics and bootstrap CIs.\n",
    )
    print(f"Wrote {len(new_rows)} new VLM rows to {results_long_path}")


if __name__ == "__main__":
    main()
