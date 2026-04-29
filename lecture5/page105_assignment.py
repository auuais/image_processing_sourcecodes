from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, finalize_figure, load_gray_image, normalize_channel, plot_bgr, plot_gray, save_image


def otsu_binary(image: np.ndarray, invert: bool = False) -> tuple[float, np.ndarray]:
    threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(image, 0, 255, threshold_mode | cv2.THRESH_OTSU)


def compute_external_internal(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    external = np.zeros_like(binary)
    internal = np.zeros_like(binary)

    if hierarchy is None:
        return external, internal

    for index in range(len(contours)):
        if hierarchy[0][index][3] == -1:
            cv2.drawContours(external, contours, index, 255, -1)
        else:
            cv2.drawContours(internal, contours, index, 255, -1)

    return external, internal


def render_random_components(labelmap: np.ndarray, stats: np.ndarray, count: int = 5, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid_labels = [label for label in range(1, stats.shape[0]) if stats[label, cv2.CC_STAT_AREA] > 20]
    selected = rng.choice(valid_labels, size=min(count, len(valid_labels)), replace=False) if valid_labels else []

    colored = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in selected:
        color = tuple(int(v) for v in rng.integers(40, 256, size=3))
        colored[labelmap == label] = color

    return colored


def run_component_viewer(binary: np.ndarray, labelmap: np.ndarray, stats: np.ndarray) -> None:
    window_name = "Random components"
    state = {"seed": 0}

    while True:
        selected = render_random_components(labelmap, stats, seed=state["seed"])
        preview = np.hstack([cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), selected])
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30)
        if key == 32:
            state["seed"] += 1
        elif key == 27:
            break

    cv2.destroyWindow(window_name)


def save_assignment_images(output_dir: Path, images: dict[str, np.ndarray]) -> None:
    for name, image in images.items():
        save_image(output_dir / f"{name}.png", image)


def main(show: bool = False, interactive: bool = False) -> None:
    assignment_dir = OUTPUT_DIR / "page105_assignment"

    otsu_original = load_gray_image("coins.png")
    otsu_thr, otsu_result = otsu_binary(otsu_original, invert=False)
    print(f"Otsu threshold on coins image: {otsu_thr:.2f}")
    save_assignment_images(
        assignment_dir,
        {
            "01_otsu_original_coins": otsu_original,
            "02_otsu_result": otsu_result,
        },
    )

    contour_original = load_gray_image("bnw_shapes.png")
    external, internal = compute_external_internal(contour_original)
    save_assignment_images(
        assignment_dir,
        {
            "03_contours_original_shapes": contour_original,
            "04_contours_external": external,
            "05_contours_internal": internal,
        },
    )

    component_original = load_gray_image("bnw_shapes.png")
    component_binary = component_original.copy()
    num_labels, labelmap, stats, _centers = cv2.connectedComponentsWithStats(component_binary, connectivity=8, ltype=cv2.CV_32S)
    print(f"Connected components found in shapes image: {num_labels - 1}")
    random_components = render_random_components(labelmap, stats, seed=0)
    save_assignment_images(
        assignment_dir,
        {
            "06_components_original_shapes": component_original,
            "07_components_random_five": random_components,
        },
    )

    distance_original = load_gray_image("distance_circles.png")
    distance = cv2.distanceTransform(distance_original, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_result = normalize_channel(distance)
    save_assignment_images(
        assignment_dir,
        {
            "08_distance_original_shapes": distance_original,
            "09_distance_transform": distance_result,
        },
    )

    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    views = [
        ("Otsu original", otsu_original, "gray"),
        ("Otsu result", otsu_result, "gray"),
        ("Contour original", contour_original, "gray"),
        ("External/Internal contours", np.dstack([external, internal, np.zeros_like(external)]), "bgr"),
        ("Connected-components original", component_original, "gray"),
        ("Random 5 components", random_components, "bgr"),
        ("Distance original", distance_original, "gray"),
        ("Distance transform", distance_result, "gray"),
    ]

    for ax, (title, image, mode) in zip(axes.flat, views):
        if mode == "bgr":
            plot_bgr(ax, image, title)
        else:
            plot_gray(ax, image, title)

    for ax in axes.flat[len(views):]:
        ax.axis("off")

    finalize_figure(fig, "page105_assignment_summary.png", show)

    if interactive:
        run_component_viewer(component_binary, labelmap, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 105 - Assignment solution")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    main(show=args.show, interactive=args.interactive)
