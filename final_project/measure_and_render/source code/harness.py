from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class PromptVariant:
    task: str
    sample_id: str
    mode: str
    image_path: str
    prompt: str
    expected_answer: str
    measurement_text: str


@dataclass
class MetricSummary:
    sample_count: int
    exact_accuracy: float
    mae: float
    rmse: float
    bias: float
    tolerance_accuracy: float
    mae_ci_low: float
    mae_ci_high: float


@dataclass
class TaskBenchmarkResult:
    task: str
    baseline_name: str
    scaffold_name: str
    baseline_summary: MetricSummary
    scaffold_summary: MetricSummary
    metrics_path: Path
    manifest_path: Path
    results_path: Path
    summary_plot_path: Path
    manifest_variants: list[PromptVariant]


def write_manifest(path: Path, variants: list[PromptVariant]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for variant in variants:
            handle.write(json.dumps(asdict(variant), ensure_ascii=True) + "\n")


def combine_manifests(path: Path, variants: list[PromptVariant]) -> None:
    write_manifest(path, variants)


def _bootstrap_mae(errors: np.ndarray, bootstrap_samples: int = 2000, seed: int = 20260610) -> tuple[float, float]:
    if errors.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(errors, size=errors.size, replace=True)
        draws.append(float(np.mean(np.abs(sample))))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize_numeric_predictions(
    pred_truth_pairs: list[tuple[float, float]],
    exact_tolerance: float = 0.0,
    bootstrap_samples: int = 2000,
    seed: int = 20260610,
) -> MetricSummary:
    if not pred_truth_pairs:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    predictions = np.array([pair[0] for pair in pred_truth_pairs], dtype=np.float32)
    truth = np.array([pair[1] for pair in pred_truth_pairs], dtype=np.float32)
    errors = predictions - truth
    mae_ci_low, mae_ci_high = _bootstrap_mae(errors, bootstrap_samples=bootstrap_samples, seed=seed)
    abs_errors = np.abs(errors)
    return MetricSummary(
        sample_count=len(pred_truth_pairs),
        exact_accuracy=float(np.mean(abs_errors <= 1e-6)),
        mae=float(np.mean(abs_errors)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        bias=float(np.mean(errors)),
        tolerance_accuracy=float(np.mean(abs_errors <= exact_tolerance)),
        mae_ci_low=mae_ci_low,
        mae_ci_high=mae_ci_high,
    )
