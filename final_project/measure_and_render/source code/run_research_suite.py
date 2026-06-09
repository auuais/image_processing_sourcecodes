from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from angle_task import run_angle_benchmark
from common import APPENDIX_OUTPUT_DIR, VLM_OUTPUT_DIR, ensure_base_directories, write_csv, write_text
from counting_task import run_counting_benchmark
from harness import TaskBenchmarkResult, combine_manifests


def write_measurement_appendix(results: list[TaskBenchmarkResult]) -> None:
    lines = [
        "# Measurement Quality Appendix",
        "",
        "These classical-CV numbers are a precondition for the frozen-VLM study, not the dependent variable.",
        "",
        "| Task | Baseline | Scaffold | Baseline MAE | Scaffold MAE | Baseline metric | Scaffold metric |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    rows: list[dict[str, object]] = []
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
    write_text(APPENDIX_OUTPUT_DIR / "MEASUREMENT_PRECONDITION.md", "\n".join(lines) + "\n")
    write_csv(APPENDIX_OUTPUT_DIR / "measurement_quality.csv", rows, fieldnames=list(rows[0].keys()))


def save_measurement_plot(results: list[TaskBenchmarkResult]) -> None:
    task_names = [result.task for result in results]
    baseline_mae = [result.baseline_summary.mae for result in results]
    scaffold_mae = [result.scaffold_summary.mae for result in results]

    x = np.arange(len(task_names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, baseline_mae, width=width, label="Measurement baseline")
    ax.bar(x + width / 2, scaffold_mae, width=width, label="Measurement scaffold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylabel("MAE")
    ax.set_title("Classical measurement quality precondition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(APPENDIX_OUTPUT_DIR / "measurement_quality.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_base_directories()
    counting = run_counting_benchmark()
    angle = run_angle_benchmark()
    results = [counting, angle]
    write_measurement_appendix(results)
    save_measurement_plot(results)
    combined_variants = counting.manifest_variants + angle.manifest_variants
    combine_manifests(VLM_OUTPUT_DIR / "combined_vlm_manifest.jsonl", combined_variants)

    write_text(
        VLM_OUTPUT_DIR / "RESULTS_SCOPE.txt",
        "The dependent variable is frozen-VLM task accuracy and MAE by condition. Classical measurement quality lives in output/appendix/.\n",
    )
    print("Completed research suite.")
    for result in results:
        print(f"{result.task}: baseline MAE={result.baseline_summary.mae:.3f}, scaffold MAE={result.scaffold_summary.mae:.3f}")


if __name__ == "__main__":
    main()
