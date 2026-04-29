from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUTPUT_DIR,
    bow_image_paths,
    build_bow_vocabulary,
    compute_descriptor,
    detect_surf_keypoints,
    draw_matches,
    encode_bow_histogram,
    ensure_color,
    finalize_figure,
    find_homography_from_matches,
    load_color_image,
    load_gray_image,
    parse_show_flag,
    plot_bgr,
    ratio_test_matches,
    resolved_descriptor_label,
    save_image,
    sift_descriptors,
    surf_label,
    warp_corners,
)


ASSIGNMENT_DIR = OUTPUT_DIR / "page91_assignment"


def descriptor_result(descriptor_name: str, image1, image2, gray1, gray2) -> dict[str, object]:
    points1 = detect_surf_keypoints(gray1, max_points=550)
    points2 = detect_surf_keypoints(gray2, max_points=550)
    points1, desc1 = compute_descriptor(descriptor_name, gray1, points1)
    points2, desc2 = compute_descriptor(descriptor_name, gray2, points2)
    matches = ratio_test_matches(desc1, desc2, descriptor_name, ratio=0.75)
    homography, inlier_mask = find_homography_from_matches(points1, points2, matches)
    mask_values = [int(flag) for flag in inlier_mask.ravel().tolist()] if inlier_mask is not None else None
    match_view = draw_matches(image1, points1, image2, points2, matches, max_matches=50, matches_mask=mask_values)

    projected = image2.copy()
    if homography is not None:
        corners = warp_corners(image1, homography)
        cv2.polylines(projected, [corners.astype(int)], True, (0, 255, 255), 3, cv2.LINE_AA)

    return {
        "label": resolved_descriptor_label(descriptor_name),
        "keypoints1": len(points1),
        "keypoints2": len(points2),
        "matches": len(matches),
        "inliers": int(inlier_mask.sum()) if inlier_mask is not None else 0,
        "match_view": match_view,
        "projected": projected,
    }


def bow_result() -> tuple[np.ndarray, float, dict[str, np.ndarray], dict[str, Path]]:
    records = bow_image_paths()
    descriptor_sets = []
    histograms = []
    labels = []
    sample_paths: dict[str, Path] = {}

    for label, image_path in records:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        descriptor_sets.append(sift_descriptors(gray))
        labels.append(label)
        sample_paths.setdefault(label, image_path)

    vocabulary = build_bow_vocabulary(descriptor_sets, cluster_count=32)
    for descriptors in descriptor_sets:
        histograms.append(encode_bow_histogram(descriptors, vocabulary))

    grouped = defaultdict(list)
    for label, histogram in zip(labels, histograms):
        grouped[label].append(histogram)
    class_means = {label: np.mean(items, axis=0) for label, items in grouped.items()}
    centroids = {label: np.mean(items, axis=0) for label, items in grouped.items()}
    correct = 0
    for label, histogram in zip(labels, histograms):
        predicted = min(centroids, key=lambda name: np.linalg.norm(histogram - centroids[name]))
        correct += int(predicted == label)
    accuracy = correct / max(len(labels), 1)
    return vocabulary, accuracy, class_means, sample_paths


def save_descriptor_panel(name: str, image) -> Path:
    path = ASSIGNMENT_DIR / f"{name}.png"
    return save_image(path, image)


def main(show: bool = False) -> None:
    ASSIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

    image1 = load_color_image("match_scene_reference.png")
    image2 = load_color_image("match_scene_query.png")
    gray1 = load_gray_image("match_scene_reference.png")
    gray2 = load_gray_image("match_scene_query.png")

    descriptor_outputs = []
    for descriptor_name in ["SURF", "BRIEF", "ORB"]:
        result = descriptor_result(descriptor_name, image1, image2, gray1, gray2)
        save_descriptor_panel(f"{descriptor_name.lower()}_matches", result["match_view"])
        save_descriptor_panel(f"{descriptor_name.lower()}_projection", result["projected"])
        descriptor_outputs.append(result)

    _, bow_accuracy, class_means, sample_paths = bow_result()
    bow_fig, bow_axes = plt.subplots(2, 3, figsize=(16, 9))
    ordered_labels = sorted(class_means)
    for index, label in enumerate(ordered_labels):
        sample_image = cv2.imread(str(sample_paths[label]), cv2.IMREAD_COLOR)
        plot_bgr(bow_axes[0, index], sample_image, f"{label} sample")
        bow_axes[1, index].bar(np.arange(len(class_means[label])), class_means[label], color="#1d3557")
        bow_axes[1, index].set_title(f"{label} BoW histogram")
        bow_axes[1, index].set_xlabel("Visual word")
        bow_axes[1, index].set_ylabel("Normalized count")
    bow_fig.suptitle(f"Assignment part 2 - BoW histograms, nearest-centroid accuracy: {bow_accuracy:.1%}", fontsize=14)
    bow_path = ASSIGNMENT_DIR / "bow_histograms.png"
    bow_fig.tight_layout()
    bow_fig.savefig(bow_path, dpi=160, bbox_inches="tight")
    plt.close(bow_fig)
    print(f"Saved figure: {bow_path}")

    summary_fig, summary_axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_bgr(summary_axes[0, 0], descriptor_outputs[0]["match_view"], descriptor_outputs[0]["label"])
    plot_bgr(summary_axes[0, 1], descriptor_outputs[1]["match_view"], descriptor_outputs[1]["label"])
    plot_bgr(summary_axes[1, 0], descriptor_outputs[2]["match_view"], descriptor_outputs[2]["label"])
    summary_axes[1, 1].axis("off")
    lines = [f"SURF detector backend: {surf_label()}", f"Part 2 BoW accuracy: {bow_accuracy:.1%}"]
    for result in descriptor_outputs:
        lines.append(f"{result['label']}: matches={result['matches']}, inliers={result['inliers']}")
    summary_axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    summary_axes[1, 1].set_title("Assignment summary")
    finalize_figure(summary_fig, "page91_assignment_summary.png", show=show)


if __name__ == "__main__":
    main(show=parse_show_flag("Lecture 8 page 91 assignment"))
