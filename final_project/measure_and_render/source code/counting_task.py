from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from baselines import counting_prompt, format_count_text, render_counting_som_overlay, save_variant_image
from common import APPENDIX_OUTPUT_DIR, CountingSample, TRUST_DELTA_COUNTING, add_measurement_footer, ensure_task_output_dirs, finalize_figure, load_color_image, palette_bgr, plot_bgr, plot_gray, read_counting_metadata, render_coordinate_grid, slugify_delta, write_csv, write_text
from generate_synthetic_counting_data import main as generate_counting_data
from harness import PromptVariant, TaskBenchmarkResult, summarize_numeric_predictions, write_manifest


@dataclass
class CountingResult:
    binary_mask: np.ndarray
    connected_components_count: int
    watershed_count: int
    centroids: list[tuple[int, int]]
    markers: np.ndarray
    overlay_pixels_only: np.ndarray
    overlay_grid: np.ndarray
    overlay_som: np.ndarray


def segment_objects(image: np.ndarray) -> tuple[np.ndarray, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, binary_mask = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned)
    foreground_components = sum(int(area >= 400) for area in stats[1:, cv2.CC_STAT_AREA])
    return cleaned, foreground_components


def watershed_count(image: np.ndarray, binary_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    local_max = cv2.dilate(dist_transform, peak_kernel)
    peak_threshold = max(8.0, 0.16 * float(dist_transform.max()))
    peak_mask = np.uint8((dist_transform >= local_max - 1e-6) & (dist_transform > peak_threshold)) * 255
    peak_mask = cv2.dilate(peak_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    _marker_count, peak_markers = cv2.connectedComponents(peak_mask)
    markers = peak_markers + 1
    markers[binary_mask == 0] = 1
    markers[(binary_mask > 0) & (peak_markers == 0)] = 0
    markers = cv2.watershed(image.copy(), markers)

    centroids: list[tuple[int, int]] = []
    for label in sorted(label for label in np.unique(markers) if label > 1):
        region_mask = np.uint8(markers == label)
        area = int(region_mask.sum())
        if area < 350:
            continue
        moments = cv2.moments(region_mask)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        centroids.append((center_x, center_y))
    return markers, centroids


def render_pixels_overlay(image: np.ndarray, centroids: list[tuple[int, int]], count_value: int, include_total: bool) -> np.ndarray:
    overlay = image.copy()
    colors = palette_bgr()
    for index, (center_x, center_y) in enumerate(centroids, start=1):
        color = colors[(index - 1) % len(colors)]
        cv2.circle(overlay, (center_x, center_y), 17, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (center_x, center_y), 17, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(index), (center_x - 8, center_y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if include_total:
        cv2.putText(overlay, f"count={count_value}", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 3, cv2.LINE_AA)
    return overlay


def markers_to_color(markers: np.ndarray) -> np.ndarray:
    label_ids = np.unique(markers)
    label_ids = label_ids[label_ids > 1]
    colored = np.full((*markers.shape, 3), 245, dtype=np.uint8)
    colors = palette_bgr()
    for index, label_id in enumerate(label_ids):
        colored[markers == label_id] = colors[index % len(colors)]
    colored[markers == -1] = (30, 30, 30)
    return colored


def count_sample(sample: CountingSample) -> CountingResult:
    image = load_color_image(sample.image_path)
    binary_mask, cc_count = segment_objects(image)
    markers, centroids = watershed_count(image, binary_mask)
    watershed_total = len(centroids)

    pixels_overlay = render_pixels_overlay(image, centroids, watershed_total, include_total=True)
    som_overlay = render_counting_som_overlay(image, centroids)
    grid_overlay = render_coordinate_grid(image)

    return CountingResult(
        binary_mask=binary_mask,
        connected_components_count=cc_count,
        watershed_count=watershed_total,
        centroids=centroids,
        markers=markers,
        overlay_pixels_only=pixels_overlay,
        overlay_grid=grid_overlay,
        overlay_som=som_overlay,
    )


def save_sample_figure(per_sample_dir: Path, sample: CountingSample, image: np.ndarray, result: CountingResult) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_bgr(axes[0, 0], image, f"{sample.sample_id}: raw image")
    plot_gray(axes[0, 1], result.binary_mask, f"Binary mask (CC count={result.connected_components_count})")
    plot_bgr(axes[1, 0], markers_to_color(result.markers), f"Watershed labels (count={result.watershed_count})")
    plot_bgr(axes[1, 1], result.overlay_pixels_only, "Pixels scaffold")
    finalize_figure(fig, per_sample_dir / f"{sample.sample_id}_summary.png")


def write_prompt_file(text_prompt_dir: Path, sample_id: str, condition: str, prompt: str) -> None:
    safe_condition = condition.replace("+", "p").replace("-", "m")
    write_text(text_prompt_dir / f"{sample_id}_{safe_condition}.txt", prompt + "\n")


def build_prompt_variants(variants_dir: Path, sample: CountingSample, image: np.ndarray, result: CountingResult) -> list[PromptVariant]:
    raw_path = save_variant_image(variants_dir, "raw", sample.sample_id, image)
    grid_path = save_variant_image(variants_dir, "grid", sample.sample_id, result.overlay_grid)
    som_path = save_variant_image(variants_dir, "som", sample.sample_id, result.overlay_som)
    pixels_path = save_variant_image(variants_dir, "pixels", sample.sample_id, result.overlay_pixels_only)

    measured_value = int(result.watershed_count)
    measured_text = format_count_text(measured_value)
    both_image = add_measurement_footer(result.overlay_pixels_only, measured_text)
    both_path = save_variant_image(variants_dir, "both", sample.sample_id, both_image)

    variants = [
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "raw", "raw", str(raw_path), counting_prompt("raw"), str(sample.true_count), float(sample.true_count), 0.0, "", None, 0.0, "count"),
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "cot", "cot", str(raw_path), counting_prompt("cot"), str(sample.true_count), float(sample.true_count), 0.0, "", None, 0.0, "count"),
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "grid", "grid", str(grid_path), counting_prompt("grid"), str(sample.true_count), float(sample.true_count), 0.0, "", None, 0.0, "count"),
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "som", "som", str(som_path), counting_prompt("som"), str(sample.true_count), float(sample.true_count), 0.0, "", None, 0.0, "count"),
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "pixels", "pixels", str(pixels_path), counting_prompt("pixels"), str(sample.true_count), float(sample.true_count), 0.0, measured_text, float(measured_value), 0.0, "count"),
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "text", "text", str(raw_path), counting_prompt("text", measured_text), str(sample.true_count), float(sample.true_count), 0.0, measured_text, float(measured_value), 0.0, "count"),
        PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, "both", "both", str(both_path), counting_prompt("both", measured_text), str(sample.true_count), float(sample.true_count), 0.0, measured_text, float(measured_value), 0.0, "count"),
    ]

    write_prompt_file(variants_dir / "text_prompts", sample.sample_id, "text", counting_prompt("text", measured_text))
    write_prompt_file(variants_dir / "text_prompts", sample.sample_id, "both", counting_prompt("both", measured_text))

    for delta in TRUST_DELTA_COUNTING:
        injected_value = max(0, measured_value + int(delta))
        perturbation_text = format_count_text(injected_value)
        suffix = f"delta_{slugify_delta(delta)}"

        perturbed_pixels = render_pixels_overlay(image, result.centroids, injected_value, include_total=True)
        perturbed_pixels_path = save_variant_image(variants_dir, "pixels", sample.sample_id, perturbed_pixels, suffix=suffix)
        perturbed_both_image = add_measurement_footer(perturbed_pixels, perturbation_text)
        perturbed_both_path = save_variant_image(variants_dir, "both", sample.sample_id, perturbed_both_image, suffix=suffix)

        pixels_condition = f"pixels_delta_{slugify_delta(delta)}"
        text_condition = f"text_delta_{slugify_delta(delta)}"
        both_condition = f"both_delta_{slugify_delta(delta)}"
        text_prompt = counting_prompt("text", perturbation_text)
        both_prompt = counting_prompt("both", perturbation_text)

        variants.extend(
            [
                PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, pixels_condition, "pixels", str(perturbed_pixels_path), counting_prompt("pixels"), str(sample.true_count), float(sample.true_count), 0.0, perturbation_text, float(injected_value), float(delta), "count"),
                PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, text_condition, "text", str(raw_path), text_prompt, str(sample.true_count), float(sample.true_count), 0.0, perturbation_text, float(injected_value), float(delta), "count"),
                PromptVariant("counting", sample.sample_id, sample.split, sample.source_dataset, sample.overlap_level, sample.notes, both_condition, "both", str(perturbed_both_path), both_prompt, str(sample.true_count), float(sample.true_count), 0.0, perturbation_text, float(injected_value), float(delta), "count"),
            ]
        )
        write_prompt_file(variants_dir / "text_prompts", sample.sample_id, text_condition, text_prompt)
        write_prompt_file(variants_dir / "text_prompts", sample.sample_id, both_condition, both_prompt)

    return variants


def write_results_markdown(path: Path, rows: list[dict[str, object]], baseline_summary, scaffold_summary) -> None:
    lines = [
        "# Counting Measurement Quality",
        "",
        "This appendix reports classical measurement quality, which is only a precondition for the frozen-VLM study.",
        "",
        "## Aggregate results",
        "",
        f"- Baseline (`connected_components`) exact accuracy: {baseline_summary.exact_accuracy:.2%}",
        f"- Baseline MAE: {baseline_summary.mae:.3f} (95% CI: {baseline_summary.mae_ci_low:.3f}, {baseline_summary.mae_ci_high:.3f})",
        f"- Scaffold (`watershed`) exact accuracy: {scaffold_summary.exact_accuracy:.2%}",
        f"- Scaffold MAE: {scaffold_summary.mae:.3f} (95% CI: {scaffold_summary.mae_ci_low:.3f}, {scaffold_summary.mae_ci_high:.3f})",
        "",
        "## Per-sample results",
        "",
        "| Sample | Ground truth | CC count | Watershed count | Overlap level | Notes |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['true_count']} | {row['connected_components_count']} | "
            f"{row['watershed_count']} | {row['overlap_level']} | {row['notes']} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def save_summary_plot(path: Path, rows: list[dict[str, object]]) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    truth = np.array([int(row["true_count"]) for row in rows])
    cc = np.array([int(row["connected_components_count"]) for row in rows])
    ws = np.array([int(row["watershed_count"]) for row in rows])

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(x, truth, marker="o", linewidth=2.5, label="Ground truth")
    ax.plot(x, cc, marker="s", linewidth=1.7, label="Connected components")
    ax.plot(x, ws, marker="^", linewidth=1.7, label="Watershed scaffold")
    ax.set_xticks(x[:: max(1, len(rows) // 16)])
    ax.set_xticklabels(sample_ids[:: max(1, len(rows) // 16)], rotation=30, ha="right")
    ax.set_ylabel("Object count")
    ax.set_title("Counting measurement quality precondition")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finalize_figure(fig, path)


def run_counting_benchmark() -> TaskBenchmarkResult:
    generate_counting_data()
    paths = ensure_task_output_dirs("counting")
    samples = read_counting_metadata()

    metric_rows: list[dict[str, object]] = []
    manifest_variants: list[PromptVariant] = []
    baseline_pairs: list[tuple[float, float]] = []
    scaffold_pairs: list[tuple[float, float]] = []

    for sample in samples:
        image = load_color_image(sample.image_path)
        result = count_sample(sample)
        save_sample_figure(paths["per_sample_dir"], sample, image, result)
        manifest_variants.extend(build_prompt_variants(paths["variants_dir"], sample, image, result))

        baseline_pairs.append((float(result.connected_components_count), float(sample.true_count)))
        scaffold_pairs.append((float(result.watershed_count), float(sample.true_count)))
        metric_rows.append(
            {
                "sample_id": sample.sample_id,
                "true_count": sample.true_count,
                "connected_components_count": result.connected_components_count,
                "watershed_count": result.watershed_count,
                "overlap_level": sample.overlap_level,
                "notes": sample.notes,
            }
        )

    write_csv(paths["metrics_path"], metric_rows, fieldnames=["sample_id", "true_count", "connected_components_count", "watershed_count", "overlap_level", "notes"])
    write_manifest(paths["manifest_path"], manifest_variants)

    baseline_summary = summarize_numeric_predictions(baseline_pairs, exact_tolerance=0.0)
    scaffold_summary = summarize_numeric_predictions(scaffold_pairs, exact_tolerance=0.0)
    save_summary_plot(paths["summary_plot_path"], metric_rows)
    write_results_markdown(paths["results_path"], metric_rows, baseline_summary, scaffold_summary)

    return TaskBenchmarkResult(
        task="counting",
        baseline_name="connected_components",
        scaffold_name="watershed_scaffold",
        baseline_summary=baseline_summary,
        scaffold_summary=scaffold_summary,
        metrics_path=paths["metrics_path"],
        manifest_path=paths["manifest_path"],
        results_path=paths["results_path"],
        summary_plot_path=paths["summary_plot_path"],
        manifest_variants=manifest_variants,
    )


def write_measurement_quality_appendix(result: TaskBenchmarkResult) -> None:
    APPENDIX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text(
        APPENDIX_OUTPUT_DIR / "measurement_quality_note.txt",
        "Classical measurement quality is a precondition for the VLM study, not the dependent variable.\n",
    )


def main() -> None:
    result = run_counting_benchmark()
    print(f"Task: {result.task}")
    print(f"Baseline MAE: {result.baseline_summary.mae:.3f}")
    print(f"Scaffold MAE: {result.scaffold_summary.mae:.3f}")
    print(f"Wrote metrics to: {result.metrics_path}")


if __name__ == "__main__":
    main()
