from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUTPUT_DIR,
    detect_fast_keypoints,
    detect_gftt_points,
    detect_harris_points,
    detect_sift,
    ensure_color,
    finalize_figure,
    keypoints_to_points,
    load_affine_matrix,
    load_color_image,
    load_gray_image,
    parse_show_flag,
    plot_bgr,
    save_image,
    top_keypoints,
)


DETECTOR_STYLES = {
    "FAST": {"color": (0, 255, 0), "shape": "circle"},
    "Harris": {"color": (0, 0, 255), "shape": "cross"},
    "GFTT": {"color": (255, 0, 0), "shape": "square"},
    "SIFT": {"color": (0, 255, 255), "shape": "diamond"},
}


def draw_marker(image: np.ndarray, point: tuple[float, float], color: tuple[int, int, int], shape: str, size: int = 6) -> None:
    x = int(round(point[0]))
    y = int(round(point[1]))
    if shape == "circle":
        cv2.circle(image, (x, y), size, color, 1, cv2.LINE_AA)
    elif shape == "cross":
        cv2.line(image, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
        cv2.line(image, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(image, (x - size, y - size), (x + size, y + size), color, 1, cv2.LINE_AA)
    elif shape == "diamond":
        points = np.array(
            [[x, y - size], [x + size, y], [x, y + size], [x - size, y]],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], True, color, 1, cv2.LINE_AA)
    else:
        raise ValueError(f"Unknown marker shape: {shape}")


def draw_detector_overlay(image: np.ndarray, detector_points: dict[str, np.ndarray], title_text: str | None = None) -> np.ndarray:
    overlay = ensure_color(image).copy()
    for detector_name, points in detector_points.items():
        style = DETECTOR_STYLES[detector_name]
        for point in points:
            draw_marker(overlay, tuple(point), style["color"], style["shape"], size=5)

    if title_text:
        cv2.putText(
            overlay,
            title_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    legend_y = 46
    for detector_name, style in DETECTOR_STYLES.items():
        draw_marker(overlay, (20, legend_y - 4), style["color"], style["shape"], size=5)
        cv2.putText(
            overlay,
            detector_name,
            (35, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            style["color"],
            2,
            cv2.LINE_AA,
        )
        legend_y += 24

    return overlay


def transform_points(points: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    homogeneous = np.hstack([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)])
    transformed = homogeneous @ affine_matrix.T
    return transformed.astype(np.float32)


def compute_repeatability(
    points1: np.ndarray,
    points2: np.ndarray,
    affine_matrix: np.ndarray,
    image_shape: tuple[int, int],
    tolerance: float = 5.0,
) -> tuple[int, int, float]:
    if points1.size == 0 or points2.size == 0:
        return 0, 0, 0.0

    projected = transform_points(points1, affine_matrix)
    height, width = image_shape
    valid_mask = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    projected = projected[valid_mask]
    if projected.size == 0:
        return 0, 0, 0.0

    used = np.zeros(len(points2), dtype=bool)
    matched = 0
    tolerance_sq = tolerance * tolerance

    for projected_point in projected:
        deltas = points2 - projected_point
        distances_sq = np.sum(deltas * deltas, axis=1)
        candidate_index = int(np.argmin(distances_sq))
        if distances_sq[candidate_index] <= tolerance_sq and not used[candidate_index]:
            used[candidate_index] = True
            matched += 1

    repeatability = matched / len(projected)
    return matched, len(projected), repeatability


def detect_all_feature_sets(gray: np.ndarray) -> dict[str, np.ndarray]:
    fast_keypoints = top_keypoints(detect_fast_keypoints(gray, threshold=30, nonmax_suppression=True), 150)
    sift_keypoints, _ = detect_sift(gray, nfeatures=120)
    sift_keypoints = top_keypoints(sift_keypoints, 120)

    return {
        "FAST": keypoints_to_points(fast_keypoints),
        "Harris": detect_harris_points(gray, max_corners=120, quality_level=0.01, min_distance=5),
        "GFTT": detect_gftt_points(gray, max_corners=120, quality_level=0.03, min_distance=6),
        "SIFT": keypoints_to_points(sift_keypoints),
    }


def save_assignment_images(output_dir: Path, images: dict[str, np.ndarray]) -> None:
    for name, image in images.items():
        save_image(output_dir / f"{name}.png", image)


def main(show: bool = False) -> None:
    assignment_dir = OUTPUT_DIR / "page74_assignment"
    assignment_dir.mkdir(parents=True, exist_ok=True)

    image1 = load_color_image("scene01.png")
    image2 = load_color_image("scene02_moved.png")
    gray1 = load_gray_image("scene01.png")
    gray2 = load_gray_image("scene02_moved.png")
    affine_matrix = load_affine_matrix()

    features1 = detect_all_feature_sets(gray1)
    features2 = detect_all_feature_sets(gray2)
    overlay1 = draw_detector_overlay(image1, features1, "Scene 1")
    overlay2 = draw_detector_overlay(image2, features2, "Scene 2")

    individual_views = {}
    for detector_name in DETECTOR_STYLES:
        individual_views[f"scene1_{detector_name.lower()}"] = draw_detector_overlay(
            image1,
            {detector_name: features1[detector_name]},
            f"Scene 1 - {detector_name}",
        )
        individual_views[f"scene2_{detector_name.lower()}"] = draw_detector_overlay(
            image2,
            {detector_name: features2[detector_name]},
            f"Scene 2 - {detector_name}",
        )

    counts1 = [len(features1[name]) for name in DETECTOR_STYLES]
    counts2 = [len(features2[name]) for name in DETECTOR_STYLES]

    repeatability_rows = []
    for detector_name in DETECTOR_STYLES:
        matched, projected, ratio = compute_repeatability(features1[detector_name], features2[detector_name], affine_matrix, gray2.shape)
        repeatability_rows.append((detector_name, matched, projected, ratio))
        print(f"{detector_name}: image1={len(features1[detector_name])}, image2={len(features2[detector_name])}, matched={matched}/{projected}, repeatability={ratio:.3f}")

    save_assignment_images(
        assignment_dir,
        {
            "01_scene1_original": image1,
            "02_scene2_original": image2,
            "03_scene1_all_detectors": overlay1,
            "04_scene2_all_detectors": overlay2,
            **{f"{index:02d}_{name}": image for index, (name, image) in enumerate(individual_views.items(), start=5)},
        },
    )

    detector_names = list(DETECTOR_STYLES)
    x_positions = np.arange(len(detector_names))

    fig_counts, ax_counts = plt.subplots(figsize=(8, 5))
    ax_counts.bar(x_positions - 0.18, counts1, width=0.36, label="Scene 1")
    ax_counts.bar(x_positions + 0.18, counts2, width=0.36, label="Scene 2")
    ax_counts.set_xticks(x_positions)
    ax_counts.set_xticklabels(detector_names)
    ax_counts.set_ylabel("Detected keypoints")
    ax_counts.set_title("Detector counts on the two scenes")
    ax_counts.legend()
    finalize_figure(fig_counts, "page74_assignment_detector_counts.png", show=False)

    repeatability_values = [row[3] * 100.0 for row in repeatability_rows]
    fig_repeatability, ax_repeatability = plt.subplots(figsize=(8, 5))
    colors = [np.array(DETECTOR_STYLES[name]["color"])[::-1] / 255.0 for name in detector_names]
    ax_repeatability.bar(detector_names, repeatability_values, color=colors)
    ax_repeatability.set_ylim(0, 100)
    ax_repeatability.set_ylabel("Repeatability (%)")
    ax_repeatability.set_title("Repeatability from scene 1 to scene 2")
    for index, (_, matched, projected, ratio) in enumerate(repeatability_rows):
        ax_repeatability.text(index, ratio * 100.0 + 2.0, f"{matched}/{projected}", ha="center", va="bottom", fontsize=9)
    finalize_figure(fig_repeatability, "page74_assignment_repeatability.png", show=False)

    fig_grid, axes_grid = plt.subplots(2, 4, figsize=(16, 8))
    for ax, detector_name in zip(axes_grid[0], detector_names):
        plot_bgr(ax, individual_views[f"scene1_{detector_name.lower()}"], f"Scene 1 - {detector_name}")
    for ax, detector_name in zip(axes_grid[1], detector_names):
        plot_bgr(ax, individual_views[f"scene2_{detector_name.lower()}"], f"Scene 2 - {detector_name}")
    finalize_figure(fig_grid, "page74_assignment_detector_grid.png", show=False)

    fig_summary, axes_summary = plt.subplots(2, 2, figsize=(14, 10))
    plot_bgr(axes_summary[0, 0], overlay1, "Scene 1 - all detectors")
    plot_bgr(axes_summary[0, 1], overlay2, "Scene 2 - all detectors")

    axes_summary[1, 0].bar(x_positions - 0.18, counts1, width=0.36, label="Scene 1")
    axes_summary[1, 0].bar(x_positions + 0.18, counts2, width=0.36, label="Scene 2")
    axes_summary[1, 0].set_xticks(x_positions)
    axes_summary[1, 0].set_xticklabels(detector_names)
    axes_summary[1, 0].set_ylabel("Count")
    axes_summary[1, 0].set_title("Detector counts")
    axes_summary[1, 0].legend()

    axes_summary[1, 1].bar(detector_names, repeatability_values, color=colors)
    axes_summary[1, 1].set_ylim(0, 100)
    axes_summary[1, 1].set_ylabel("Repeatability (%)")
    axes_summary[1, 1].set_title("Scene 1 -> Scene 2 repeatability")
    for index, (_, matched, projected, ratio) in enumerate(repeatability_rows):
        axes_summary[1, 1].text(index, ratio * 100.0 + 2.0, f"{matched}/{projected}", ha="center", va="bottom", fontsize=9)

    finalize_figure(fig_summary, "page74_assignment_summary.png", show=show)


if __name__ == "__main__":
    main(parse_show_flag("Page 74 - Assignment solution"))
