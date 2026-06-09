from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from angle_task import run_angle_benchmark
from common import SUITE_OUTPUT_DIR, ensure_base_directories, write_csv, write_text
from counting_task import run_counting_benchmark
from harness import TaskBenchmarkResult, combine_manifests


def write_suite_markdown(results: list[TaskBenchmarkResult]) -> None:
    lines = [
        "# Measure-and-Render Research Suite",
        "",
        "This suite aggregates the current local benchmarks for the project.",
        "",
        "## Task summary",
        "",
        "| Task | Baseline | Scaffold | Baseline MAE | Scaffold MAE | Baseline metric | Scaffold metric |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for result in results:
        if result.task == "counting":
            baseline_metric = f"{result.baseline_summary.exact_accuracy:.2%} exact"
            scaffold_metric = f"{result.scaffold_summary.exact_accuracy:.2%} exact"
        else:
            baseline_metric = f"{result.baseline_summary.tolerance_accuracy:.2%} within 1 deg"
            scaffold_metric = f"{result.scaffold_summary.tolerance_accuracy:.2%} within 1 deg"
        lines.append(
            f"| {result.task} | {result.baseline_name} | {result.scaffold_name} | "
            f"{result.baseline_summary.mae:.3f} | {result.scaffold_summary.mae:.3f} | "
            f"{baseline_metric} | {scaffold_metric} |"
        )

    lines.extend(
        [
            "",
            "## Research notes",
            "",
            "- Counting benchmark already validates the strongest local claim: a content-adaptive classical measurement scaffold can substantially outperform a weaker vision baseline on precise enumeration.",
            "- Angle benchmark broadens the method beyond counting and shows that the same measure-and-render pattern transfers to geometric measurement.",
            "- The suite now emits matched `raw`, `pixels_only`, `text_only`, `both`, and `grid` variants, so the next research step is a real VLM channel study rather than more local benchmark plumbing.",
        ]
    )

    write_text(SUITE_OUTPUT_DIR / "RESEARCH_SUMMARY.md", "\n".join(lines) + "\n")


def save_suite_plot(results: list[TaskBenchmarkResult]) -> None:
    task_names = [result.task for result in results]
    baseline_mae = [result.baseline_summary.mae for result in results]
    scaffold_mae = [result.scaffold_summary.mae for result in results]

    x = np.arange(len(task_names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, baseline_mae, width=width, label="Baseline")
    ax.bar(x + width / 2, scaffold_mae, width=width, label="Scaffold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylabel("MAE")
    ax.set_title("Baseline vs scaffold MAE by task")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SUITE_OUTPUT_DIR / "suite_mae_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_suite_csv(results: list[TaskBenchmarkResult]) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "task": result.task,
                "baseline_name": result.baseline_name,
                "scaffold_name": result.scaffold_name,
                "baseline_mae": f"{result.baseline_summary.mae:.4f}",
                "scaffold_mae": f"{result.scaffold_summary.mae:.4f}",
                "baseline_rmse": f"{result.baseline_summary.rmse:.4f}",
                "scaffold_rmse": f"{result.scaffold_summary.rmse:.4f}",
                "baseline_exact_accuracy": f"{result.baseline_summary.exact_accuracy:.4f}",
                "scaffold_exact_accuracy": f"{result.scaffold_summary.exact_accuracy:.4f}",
                "baseline_tolerance_accuracy": f"{result.baseline_summary.tolerance_accuracy:.4f}",
                "scaffold_tolerance_accuracy": f"{result.scaffold_summary.tolerance_accuracy:.4f}",
                "baseline_mae_ci_low": f"{result.baseline_summary.mae_ci_low:.4f}",
                "baseline_mae_ci_high": f"{result.baseline_summary.mae_ci_high:.4f}",
                "scaffold_mae_ci_low": f"{result.scaffold_summary.mae_ci_low:.4f}",
                "scaffold_mae_ci_high": f"{result.scaffold_summary.mae_ci_high:.4f}",
            }
        )
    write_csv(
        SUITE_OUTPUT_DIR / "suite_summary.csv",
        rows,
        fieldnames=list(rows[0].keys()),
    )


def main() -> None:
    ensure_base_directories()
    counting = run_counting_benchmark()
    angle = run_angle_benchmark()
    results = [counting, angle]
    write_suite_csv(results)
    write_suite_markdown(results)
    save_suite_plot(results)
    combined_variants = counting.manifest_variants + angle.manifest_variants
    combine_manifests(SUITE_OUTPUT_DIR / "combined_vlm_manifest.jsonl", combined_variants)

    print("Completed research suite.")
    for result in results:
        print(f"{result.task}: baseline MAE={result.baseline_summary.mae:.3f}, scaffold MAE={result.scaffold_summary.mae:.3f}")


if __name__ == "__main__":
    main()
