from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import VLM_OUTPUT_DIR
from harness import summarize_numeric_predictions


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_abs_delta_summary(long_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = [
        row
        for row in long_rows
        if row["base_condition"] in {"pixels", "text", "both"}
        and row["split"] == "synthetic"
        and abs(float(row["perturbation"])) > 1e-9
    ]
    grouped: dict[tuple[str, str, str, str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        grouped[
            (
                row["task"],
                row["split"],
                row["source_dataset"],
                row["model_name"],
                row["model_id"],
                row["base_condition"],
                abs(float(row["perturbation"])),
            )
        ].append(row)

    summary_rows: list[dict[str, str]] = []
    for key, rows in sorted(grouped.items()):
        task, split, source_dataset, model_name, model_id, base_condition, abs_delta = key
        parsed_pairs: list[tuple[float, float]] = []
        follow_values: list[float] = []
        parse_values: list[int] = []
        for row in rows:
            parse_ok = int(row["parse_ok"])
            parse_values.append(parse_ok)
            if parse_ok and row["predicted_value"] != "":
                parsed_pairs.append((float(row["predicted_value"]), float(row["expected_value"])))
            if abs(float(row["perturbation"])) > 1e-9:
                follow_values.append(float(row["follow_injected"]))
        tolerance = float(rows[0]["tolerance"])
        metrics = summarize_numeric_predictions(parsed_pairs, exact_tolerance=tolerance)
        summary_rows.append(
            {
                "task": task,
                "split": split,
                "source_dataset": source_dataset,
                "model_name": model_name,
                "model_id": model_id,
                "base_condition": base_condition,
                "abs_perturbation": f"{abs_delta:.1f}",
                "sample_count": str(metrics.sample_count),
                "parse_rate": f"{float(np.mean(parse_values)) if parse_values else 0.0:.4f}",
                "exact_accuracy": f"{metrics.exact_accuracy:.4f}",
                "tolerance_accuracy": f"{metrics.tolerance_accuracy:.4f}",
                "mae": f"{metrics.mae:.4f}",
                "rmse": f"{metrics.rmse:.4f}",
                "bias": f"{metrics.bias:.4f}",
                "mae_ci_low": f"{metrics.mae_ci_low:.4f}",
                "mae_ci_high": f"{metrics.mae_ci_high:.4f}",
                "follow_rate": f"{float(np.mean(follow_values)) if follow_values else 0.0:.4f}",
            }
        )
    return summary_rows


def save_trust_plots(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not summary_rows:
        return
    by_task_model: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        by_task_model[(row["task"], row["model_name"])].append(row)

    for (task, model_name), rows in sorted(by_task_model.items()):
        abs_deltas = sorted({float(row["abs_perturbation"]) for row in rows})
        if not abs_deltas:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
        for channel in ["pixels", "text", "both"]:
            channel_rows = [row for row in rows if row["base_condition"] == channel]
            accuracy_values = []
            follow_values = []
            for abs_delta in abs_deltas:
                match = next((row for row in channel_rows if abs(float(row["abs_perturbation"]) - abs_delta) < 1e-9), None)
                accuracy_values.append(np.nan if match is None else float(match["tolerance_accuracy"]))
                follow_values.append(np.nan if match is None else float(match["follow_rate"]))
            axes[0].plot(abs_deltas, accuracy_values, marker="o", linewidth=2, label=channel)
            axes[1].plot(abs_deltas, follow_values, marker="o", linewidth=2, label=channel)
        axes[0].set_title(f"{task}: accuracy vs |delta|\n{model_name}")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True, alpha=0.25)
        axes[1].set_title(f"{task}: follow-rate vs |delta|\n{model_name}")
        axes[1].set_ylabel("Follow-rate")
        axes[1].grid(True, alpha=0.25)
        for axis in axes:
            axis.set_xlabel("Absolute injected perturbation")
            axis.set_ylim(0.0, 1.0)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_{model_name}_trust_under_error.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def write_trust_summary(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with (output_dir / "trust_under_error_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    long_path = VLM_OUTPUT_DIR / "results_long.csv"
    if not long_path.exists():
        raise FileNotFoundError("Run run_vlm_eval.py first so output/vlm/results_long.csv exists.")
    long_rows = load_csv(long_path)
    summary_rows = build_abs_delta_summary(long_rows)
    save_trust_plots(summary_rows, VLM_OUTPUT_DIR)
    write_trust_summary(summary_rows, VLM_OUTPUT_DIR)
    print(f"Wrote trust-under-error outputs to {VLM_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
