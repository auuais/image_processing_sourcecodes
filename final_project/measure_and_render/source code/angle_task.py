from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from baselines import angle_prompt, format_angle_text, render_angle_som_overlay, save_variant_image
from common import AngleSample, TRUST_DELTA_ANGLE, add_measurement_footer, ensure_task_output_dirs, finalize_figure, load_color_image, plot_bgr, read_angle_metadata, render_coordinate_grid, slugify_delta, write_csv, write_text
from generate_synthetic_angle_data import main as generate_angle_data
from harness import PromptVariant, TaskBenchmarkResult, summarize_numeric_predictions, write_manifest


@dataclass
class CandidateLine:
    p1: tuple[int, int]
    p2: tuple[int, int]
    length: float
    orientation_deg: float
    center_distance: float


@dataclass
class FittedRay:
    vertex: np.ndarray
    direction: np.ndarray
    angle_deg: float
    length: float


@dataclass
class AngleResult:
    edges: np.ndarray
    candidate_visual: np.ndarray
    baseline_angle_deg: float
    scaffold_angle_deg: float
    scaffold_ray_a: FittedRay
    scaffold_ray_b: FittedRay
    overlay_baseline: np.ndarray
    overlay_pixels_only: np.ndarray
    overlay_grid: np.ndarray
    overlay_som: np.ndarray


def normalize_angle_180(angle_deg: float) -> float:
    return float(angle_deg % 180.0)


def orientation_distance(angle_a: float, angle_b: float) -> float:
    difference = abs(angle_a - angle_b) % 180.0
    return float(min(difference, 180.0 - difference))


def circular_distance(angle_a: float, angle_b: float) -> float:
    return float(abs((angle_a - angle_b + 180.0) % 360.0 - 180.0))


def cross_2d(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0])


def point_line_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    line_vec = p2 - p1
    norm = float(np.linalg.norm(line_vec))
    if norm < 1e-6:
        return float(np.linalg.norm(point - p1))
    return float(abs(np.cross(line_vec, point - p1)) / norm)


def detect_candidate_lines(image: np.ndarray) -> tuple[np.ndarray, list[CandidateLine]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    primary_mask = cv2.inRange(gray, 0, 95)
    kernel = np.ones((3, 3), dtype=np.uint8)
    primary_mask = cv2.morphologyEx(primary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    lines = cv2.HoughLinesP(primary_mask, 1, np.pi / 180.0, threshold=30, minLineLength=50, maxLineGap=20)

    height, width = image.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    candidates: list[CandidateLine] = []

    if lines is not None:
        for entry in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, entry)
            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)
            length = float(np.linalg.norm(p2 - p1))
            if length < 50.0:
                continue
            center_distance = point_line_distance(center, p1, p2)
            endpoint_distance = min(float(np.linalg.norm(p1 - center)), float(np.linalg.norm(p2 - center)))
            if center_distance > 165.0 and endpoint_distance > 260.0:
                continue
            candidates.append(
                CandidateLine(
                    p1=(x1, y1),
                    p2=(x2, y2),
                    length=length,
                    orientation_deg=normalize_angle_180(np.degrees(np.arctan2(y2 - y1, x2 - x1))),
                    center_distance=center_distance,
                )
            )
    return primary_mask, candidates


def densest_cluster_center(points: np.ndarray, radius: float) -> np.ndarray:
    best_index = 0
    best_count = -1
    for index, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        count = int(np.sum(distances <= radius))
        if count > best_count:
            best_index = index
            best_count = count
    cluster_center = points[np.linalg.norm(points - points[best_index], axis=1) <= radius].mean(axis=0)
    return cluster_center.astype(np.float32)


def line_intersection(line_a: CandidateLine, line_b: CandidateLine) -> np.ndarray | None:
    p = np.array(line_a.p1, dtype=np.float32)
    r = np.array(line_a.p2, dtype=np.float32) - p
    q = np.array(line_b.p1, dtype=np.float32)
    s = np.array(line_b.p2, dtype=np.float32) - q
    denominator = cross_2d(r, s)
    if abs(denominator) < 1e-6:
        return None
    t = cross_2d(q - p, s) / denominator
    return (p + t * r).astype(np.float32)


def estimate_vertex(lines: list[CandidateLine], image_shape: tuple[int, ...], radius: float = 34.0) -> np.ndarray:
    height, width = image_shape[:2]
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    ranked = sorted(lines, key=lambda item: item.length, reverse=True)[:24]

    intersections: list[np.ndarray] = []
    for index, line_a in enumerate(ranked):
        for line_b in ranked[index + 1 :]:
            if orientation_distance(line_a.orientation_deg, line_b.orientation_deg) < 15.0:
                continue
            point = line_intersection(line_a, line_b)
            if point is None:
                continue
            if -40.0 <= point[0] <= width + 40.0 and -40.0 <= point[1] <= height + 40.0:
                intersections.append(point)

    if intersections:
        points = np.array(intersections, dtype=np.float32)
        near_center = np.linalg.norm(points - image_center, axis=1) <= 140.0
        if np.any(near_center):
            points = points[near_center]
        return densest_cluster_center(points, radius)

    endpoints = np.array([line.p1 for line in ranked] + [line.p2 for line in ranked], dtype=np.float32)
    return densest_cluster_center(endpoints, radius)


def directed_angle_from_vertex(line: CandidateLine, vertex: np.ndarray) -> tuple[float, float]:
    p1 = np.array(line.p1, dtype=np.float32)
    p2 = np.array(line.p2, dtype=np.float32)
    dist1 = float(np.linalg.norm(p1 - vertex))
    dist2 = float(np.linalg.norm(p2 - vertex))
    far_point = p1 if dist1 >= dist2 else p2
    direction = far_point - vertex
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0)
    ray_length = max(dist1, dist2)
    return angle_deg, ray_length


def cluster_ray_directions(lines: list[CandidateLine], vertex: np.ndarray, angle_threshold: float = 12.0) -> list[list[tuple[CandidateLine, float, float]]]:
    enriched = [(line, *directed_angle_from_vertex(line, vertex)) for line in lines]
    clusters: list[list[tuple[CandidateLine, float, float]]] = []
    for item in enriched:
        _, angle_deg, _ = item
        matched = False
        for cluster in clusters:
            cluster_angles = [entry[1] for entry in cluster]
            mean_angle = circular_mean(cluster_angles)
            if circular_distance(angle_deg, mean_angle) <= angle_threshold:
                cluster.append(item)
                matched = True
                break
        if not matched:
            clusters.append([item])
    return clusters


def circular_mean(angles_deg: list[float]) -> float:
    radians = np.deg2rad(angles_deg)
    mean_x = float(np.mean(np.cos(radians)))
    mean_y = float(np.mean(np.sin(radians)))
    return float(np.degrees(np.arctan2(mean_y, mean_x)) % 360.0)


def cluster_to_ray(cluster: list[tuple[CandidateLine, float, float]], vertex: np.ndarray) -> FittedRay:
    angles = [entry[1] for entry in cluster]
    lengths = [entry[2] for entry in cluster]
    mean_angle = circular_mean(angles)
    radians = np.deg2rad(mean_angle)
    direction = np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    return FittedRay(vertex=vertex.copy(), direction=direction, angle_deg=mean_angle, length=float(np.mean(lengths)))


def fit_cluster_ray(cluster: list[tuple[CandidateLine, float, float]], vertex: np.ndarray) -> FittedRay:
    points: list[tuple[int, int]] = []
    directed_angles: list[float] = []
    lengths: list[float] = []
    for line, angle_deg, ray_length in cluster:
        points.extend([line.p1, line.p2])
        directed_angles.append(angle_deg)
        lengths.append(ray_length)

    fitted = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, _, _ = cv2.fitLine(fitted, cv2.DIST_L2, 0, 0.01, 0.01)
    base_angle = float(np.degrees(np.arctan2(float(vy.item()), float(vx.item()))) % 360.0)
    mean_angle = circular_mean(directed_angles)
    alternate_angle = (base_angle + 180.0) % 360.0
    chosen_angle = base_angle if circular_distance(base_angle, mean_angle) <= circular_distance(alternate_angle, mean_angle) else alternate_angle
    radians = np.deg2rad(chosen_angle)
    direction = np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    return FittedRay(vertex=vertex.copy(), direction=direction, angle_deg=chosen_angle, length=float(np.mean(lengths)))


def pick_scaffold_rays(lines: list[CandidateLine], vertex: np.ndarray) -> tuple[FittedRay, FittedRay]:
    clusters = cluster_ray_directions(lines, vertex)
    scored = sorted(clusters, key=lambda cluster: sum(entry[0].length for entry in cluster), reverse=True)
    if len(scored) < 2:
        raise RuntimeError("Need at least two ray clusters to measure the angle.")
    return fit_cluster_ray(scored[0], vertex), fit_cluster_ray(scored[1], vertex)


def pick_baseline_rays(lines: list[CandidateLine], vertex: np.ndarray) -> tuple[FittedRay, FittedRay]:
    ranked = sorted(lines, key=lambda item: item.length, reverse=True)
    if len(ranked) < 2:
        raise RuntimeError("Need at least two candidate lines for the baseline angle.")
    primary_angle, _ = directed_angle_from_vertex(ranked[0], vertex)
    comparison_line = ranked[1]
    for candidate in ranked[1:]:
        candidate_angle, _ = directed_angle_from_vertex(candidate, vertex)
        if circular_distance(candidate_angle, primary_angle) >= 12.0:
            comparison_line = candidate
            break
    baseline_cluster_a = [(ranked[0], *directed_angle_from_vertex(ranked[0], vertex))]
    baseline_cluster_b = [(comparison_line, *directed_angle_from_vertex(comparison_line, vertex))]
    return cluster_to_ray(baseline_cluster_a, vertex), cluster_to_ray(baseline_cluster_b, vertex)


def ray_angle_difference(ray_a: FittedRay, ray_b: FittedRay) -> float:
    return circular_distance(ray_a.angle_deg, ray_b.angle_deg)


def ray_endpoint(ray: FittedRay, scale: float = 1.2) -> tuple[int, int]:
    endpoint = ray.vertex + ray.direction * ray.length * scale
    return int(round(endpoint[0])), int(round(endpoint[1]))


def draw_arc(overlay: np.ndarray, center: np.ndarray, start_angle_deg: float, end_angle_deg: float, radius: int, color: tuple[int, int, int]) -> None:
    diff = (end_angle_deg - start_angle_deg) % 360.0
    if diff > 180.0:
        start_angle_deg, end_angle_deg = end_angle_deg, start_angle_deg
        diff = (end_angle_deg - start_angle_deg) % 360.0
    steps = np.linspace(0.0, diff, 64)
    points = []
    for step in steps:
        radians = np.deg2rad(start_angle_deg + step)
        x = int(round(center[0] + np.cos(radians) * radius))
        y = int(round(center[1] + np.sin(radians) * radius))
        points.append([x, y])
    cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, color, 3, cv2.LINE_AA)


def draw_angle_overlay(image: np.ndarray, ray_a: FittedRay, ray_b: FittedRay, arc_angle_deg: float, label_value: float | None = None) -> np.ndarray:
    overlay = image.copy()
    center = ray_a.vertex
    shown_value = arc_angle_deg if label_value is None else label_value
    cv2.line(overlay, (int(center[0]), int(center[1])), ray_endpoint(ray_a), (50, 200, 50), 4, cv2.LINE_AA)
    cv2.line(overlay, (int(center[0]), int(center[1])), ray_endpoint(ray_b), (60, 140, 240), 4, cv2.LINE_AA)
    draw_arc(overlay, center, ray_a.angle_deg, ray_b.angle_deg, radius=64, color=(210, 100, 210))
    cv2.circle(overlay, (int(center[0]), int(center[1])), 8, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.putText(overlay, f"{shown_value:.1f} deg", (int(center[0]) + 24, int(center[1]) - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return overlay


def build_candidate_visual(image: np.ndarray, lines: list[CandidateLine], vertex: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for line in lines:
        cv2.line(overlay, line.p1, line.p2, (200, 120, 70), 2, cv2.LINE_AA)
    cv2.circle(overlay, (int(vertex[0]), int(vertex[1])), 8, (210, 100, 210), -1, cv2.LINE_AA)
    return overlay


def measure_angle(sample: AngleSample) -> AngleResult:
    image = load_color_image(sample.image_path)
    edges, candidates = detect_candidate_lines(image)
    if len(candidates) < 2:
        raise RuntimeError(f"Not enough candidate lines detected for {sample.sample_id}")

    vertex = estimate_vertex(candidates, image.shape)
    selected_lines = [line for line in candidates if min(np.linalg.norm(np.array(line.p1) - vertex), np.linalg.norm(np.array(line.p2) - vertex)) <= 90.0]
    if len(selected_lines) < 2 or len(cluster_ray_directions(selected_lines, vertex)) < 2:
        selected_lines = candidates

    baseline_ray_a, baseline_ray_b = pick_baseline_rays(selected_lines, vertex)
    baseline_angle_deg = ray_angle_difference(baseline_ray_a, baseline_ray_b)

    scaffold_ray_a, scaffold_ray_b = pick_scaffold_rays(selected_lines, vertex)
    scaffold_angle_deg = ray_angle_difference(scaffold_ray_a, scaffold_ray_b)

    baseline_overlay = draw_angle_overlay(image, baseline_ray_a, baseline_ray_b, baseline_angle_deg)
    pixels_overlay = draw_angle_overlay(image, scaffold_ray_a, scaffold_ray_b, scaffold_angle_deg)
    grid_overlay = render_coordinate_grid(image)
    som_overlay = render_angle_som_overlay(
        image,
        (int(scaffold_ray_a.vertex[0]), int(scaffold_ray_a.vertex[1])),
        ray_endpoint(scaffold_ray_a),
        ray_endpoint(scaffold_ray_b),
    )

    return AngleResult(
        edges=edges,
        candidate_visual=build_candidate_visual(image, selected_lines, vertex),
        baseline_angle_deg=baseline_angle_deg,
        scaffold_angle_deg=scaffold_angle_deg,
        scaffold_ray_a=scaffold_ray_a,
        scaffold_ray_b=scaffold_ray_b,
        overlay_baseline=baseline_overlay,
        overlay_pixels_only=pixels_overlay,
        overlay_grid=grid_overlay,
        overlay_som=som_overlay,
    )


def save_sample_figure(per_sample_dir: Path, sample: AngleSample, image: np.ndarray, result: AngleResult) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_bgr(axes[0, 0], image, f"{sample.sample_id}: raw image")
    plot_bgr(axes[0, 1], result.candidate_visual, "Candidate lines near inferred vertex")
    plot_bgr(axes[1, 0], result.overlay_baseline, f"Naive Hough baseline ({result.baseline_angle_deg:.1f} deg)")
    plot_bgr(axes[1, 1], result.overlay_pixels_only, f"Scaffold measurement ({result.scaffold_angle_deg:.1f} deg)")
    finalize_figure(fig, per_sample_dir / f"{sample.sample_id}_summary.png")


def write_prompt_file(text_prompt_dir: Path, sample_id: str, condition: str, prompt: str) -> None:
    safe_condition = condition.replace("+", "p").replace("-", "m")
    write_text(text_prompt_dir / f"{sample_id}_{safe_condition}.txt", prompt + "\n")


def build_prompt_variants(variants_dir: Path, sample: AngleSample, image: np.ndarray, result: AngleResult) -> list[PromptVariant]:
    raw_path = save_variant_image(variants_dir, "raw", sample.sample_id, image)
    grid_path = save_variant_image(variants_dir, "grid", sample.sample_id, result.overlay_grid)
    som_path = save_variant_image(variants_dir, "som", sample.sample_id, result.overlay_som)
    pixels_path = save_variant_image(variants_dir, "pixels", sample.sample_id, result.overlay_pixels_only)

    measured_value = float(result.scaffold_angle_deg)
    measured_text = format_angle_text(measured_value)
    both_image = add_measurement_footer(result.overlay_pixels_only, measured_text)
    both_path = save_variant_image(variants_dir, "both", sample.sample_id, both_image)

    variants = [
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "raw", "raw", str(raw_path), angle_prompt("raw"), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, "", None, 0.0, "degrees"),
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "cot", "cot", str(raw_path), angle_prompt("cot"), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, "", None, 0.0, "degrees"),
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "grid", "grid", str(grid_path), angle_prompt("grid"), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, "", None, 0.0, "degrees"),
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "som", "som", str(som_path), angle_prompt("som"), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, "", None, 0.0, "degrees"),
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "pixels", "pixels", str(pixels_path), angle_prompt("pixels"), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, measured_text, float(measured_value), 0.0, "degrees"),
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "text", "text", str(raw_path), angle_prompt("text", measured_text), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, measured_text, float(measured_value), 0.0, "degrees"),
        PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, "both", "both", str(both_path), angle_prompt("both", measured_text), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, measured_text, float(measured_value), 0.0, "degrees"),
    ]

    write_prompt_file(variants_dir / "text_prompts", sample.sample_id, "text", angle_prompt("text", measured_text))
    write_prompt_file(variants_dir / "text_prompts", sample.sample_id, "both", angle_prompt("both", measured_text))

    for delta in TRUST_DELTA_ANGLE:
        injected_value = measured_value + float(delta)
        perturbation_text = format_angle_text(injected_value)
        suffix = f"delta_{slugify_delta(delta)}"

        perturbed_pixels = draw_angle_overlay(image, result.scaffold_ray_a, result.scaffold_ray_b, result.scaffold_angle_deg, label_value=injected_value)
        perturbed_pixels_path = save_variant_image(variants_dir, "pixels", sample.sample_id, perturbed_pixels, suffix=suffix)
        perturbed_both_image = add_measurement_footer(perturbed_pixels, perturbation_text)
        perturbed_both_path = save_variant_image(variants_dir, "both", sample.sample_id, perturbed_both_image, suffix=suffix)

        pixels_condition = f"pixels_delta_{slugify_delta(delta)}"
        text_condition = f"text_delta_{slugify_delta(delta)}"
        both_condition = f"both_delta_{slugify_delta(delta)}"
        text_prompt = angle_prompt("text", perturbation_text)
        both_prompt = angle_prompt("both", perturbation_text)

        variants.extend(
            [
                PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, pixels_condition, "pixels", str(perturbed_pixels_path), angle_prompt("pixels"), f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, perturbation_text, float(injected_value), float(delta), "degrees"),
                PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, text_condition, "text", str(raw_path), text_prompt, f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, perturbation_text, float(injected_value), float(delta), "degrees"),
                PromptVariant("angle", sample.sample_id, sample.split, sample.source_dataset, sample.clutter_level, sample.notes, both_condition, "both", str(perturbed_both_path), both_prompt, f"{sample.true_angle_deg:.1f}", float(sample.true_angle_deg), 5.0, perturbation_text, float(injected_value), float(delta), "degrees"),
            ]
        )
        write_prompt_file(variants_dir / "text_prompts", sample.sample_id, text_condition, text_prompt)
        write_prompt_file(variants_dir / "text_prompts", sample.sample_id, both_condition, both_prompt)

    return variants


def write_results_markdown(path: Path, rows: list[dict[str, object]], baseline_summary, scaffold_summary) -> None:
    lines = [
        "# Angle Measurement Quality",
        "",
        "This appendix reports classical measurement quality, which is only a precondition for the frozen-VLM study.",
        "",
        "## Aggregate results",
        "",
        f"- Baseline (`naive_hough`) exact accuracy: {baseline_summary.exact_accuracy:.2%}",
        f"- Baseline within-1deg accuracy: {baseline_summary.tolerance_accuracy:.2%}",
        f"- Baseline MAE: {baseline_summary.mae:.3f} (95% CI: {baseline_summary.mae_ci_low:.3f}, {baseline_summary.mae_ci_high:.3f})",
        f"- Scaffold (`vertex_clustered_ray_scaffold`) exact accuracy: {scaffold_summary.exact_accuracy:.2%}",
        f"- Scaffold within-1deg accuracy: {scaffold_summary.tolerance_accuracy:.2%}",
        f"- Scaffold MAE: {scaffold_summary.mae:.3f} (95% CI: {scaffold_summary.mae_ci_low:.3f}, {scaffold_summary.mae_ci_high:.3f})",
        "",
        "## Per-sample results",
        "",
        "| Sample | Ground truth | Baseline angle | Scaffold angle | Clutter level | Notes |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['true_angle_deg']:.1f} | {row['baseline_angle_deg']:.1f} | "
            f"{row['scaffold_angle_deg']:.1f} | {row['clutter_level']} | {row['notes']} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def save_summary_plot(path: Path, rows: list[dict[str, object]]) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    truth = np.array([float(row["true_angle_deg"]) for row in rows])
    baseline = np.array([float(row["baseline_angle_deg"]) for row in rows])
    scaffold = np.array([float(row["scaffold_angle_deg"]) for row in rows])

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(x, truth, marker="o", linewidth=2.5, label="Ground truth")
    ax.plot(x, baseline, marker="s", linewidth=1.7, label="Naive Hough baseline")
    ax.plot(x, scaffold, marker="^", linewidth=1.7, label="Vertex-clustered ray scaffold")
    step = max(1, len(rows) // 16)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(sample_ids[::step], rotation=30, ha="right")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Angle measurement quality precondition")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finalize_figure(fig, path)


def run_angle_benchmark() -> TaskBenchmarkResult:
    generate_angle_data()
    paths = ensure_task_output_dirs("angle")
    samples = read_angle_metadata()

    metric_rows: list[dict[str, object]] = []
    manifest_variants: list[PromptVariant] = []
    baseline_pairs: list[tuple[float, float]] = []
    scaffold_pairs: list[tuple[float, float]] = []

    for sample in samples:
        image = load_color_image(sample.image_path)
        result = measure_angle(sample)
        save_sample_figure(paths["per_sample_dir"], sample, image, result)
        manifest_variants.extend(build_prompt_variants(paths["variants_dir"], sample, image, result))

        baseline_pairs.append((float(result.baseline_angle_deg), float(sample.true_angle_deg)))
        scaffold_pairs.append((float(result.scaffold_angle_deg), float(sample.true_angle_deg)))
        metric_rows.append(
            {
                "sample_id": sample.sample_id,
                "true_angle_deg": sample.true_angle_deg,
                "baseline_angle_deg": result.baseline_angle_deg,
                "scaffold_angle_deg": result.scaffold_angle_deg,
                "clutter_level": sample.clutter_level,
                "notes": sample.notes,
            }
        )

    write_csv(paths["metrics_path"], metric_rows, fieldnames=["sample_id", "true_angle_deg", "baseline_angle_deg", "scaffold_angle_deg", "clutter_level", "notes"])
    write_manifest(paths["manifest_path"], manifest_variants)

    baseline_summary = summarize_numeric_predictions(baseline_pairs, exact_tolerance=1.0)
    scaffold_summary = summarize_numeric_predictions(scaffold_pairs, exact_tolerance=1.0)
    save_summary_plot(paths["summary_plot_path"], metric_rows)
    write_results_markdown(paths["results_path"], metric_rows, baseline_summary, scaffold_summary)

    return TaskBenchmarkResult(
        task="angle",
        baseline_name="naive_hough",
        scaffold_name="vertex_clustered_ray_scaffold",
        baseline_summary=baseline_summary,
        scaffold_summary=scaffold_summary,
        metrics_path=paths["metrics_path"],
        manifest_path=paths["manifest_path"],
        results_path=paths["results_path"],
        summary_plot_path=paths["summary_plot_path"],
        manifest_variants=manifest_variants,
    )


def main() -> None:
    result = run_angle_benchmark()
    print(f"Task: {result.task}")
    print(f"Baseline MAE: {result.baseline_summary.mae:.3f}")
    print(f"Scaffold MAE: {result.scaffold_summary.mae:.3f}")
    print(f"Wrote metrics to: {result.metrics_path}")


if __name__ == "__main__":
    main()
