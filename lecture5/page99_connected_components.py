from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, normalize_channel, parse_show_flag, plot_bgr, plot_gray


def colorize_labelmap(labelmap: np.ndarray) -> np.ndarray:
    normalized = normalize_channel(labelmap)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[labelmap == 0] = 0
    return colored


def render_components_with_centers(binary_image: np.ndarray, labelmap: np.ndarray, stats: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for label in range(1, stats.shape[0]):
        if stats[label, cv2.CC_STAT_AREA] < 30:
            continue
        random_color = tuple(int(v) for v in rng.integers(80, 256, size=3))
        color[labelmap == label] = random_color
        center = tuple(int(v) for v in centers[label])
        cv2.circle(color, center, 4, (255, 255, 255), -1)

    return color


def main(show: bool = False) -> None:
    shapes = load_gray_image("bnw_shapes.png")
    num_labels_shapes, labelmap_shapes = cv2.connectedComponents(shapes, connectivity=8, ltype=cv2.CV_32S)
    print(f"Connected components in shapes image: {num_labels_shapes - 1}")

    text_image = load_gray_image("text.png")
    otsu_thr, otsu_mask = cv2.threshold(text_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print(f"Otsu threshold for text image: {otsu_thr:.2f}")
    output = cv2.connectedComponentsWithStats(otsu_mask, connectivity=8, ltype=cv2.CV_32S)
    num_labels_page, labelmap_page, stats_page, centers_page = output
    print(f"Connected components in text image: {num_labels_page - 1}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_gray(axes[0, 0], shapes, "Binary shapes")
    plot_bgr(axes[0, 1], colorize_labelmap(labelmap_shapes), "Connected components")
    plot_gray(axes[1, 0], otsu_mask, "Text + Otsu")
    plot_bgr(axes[1, 1], render_components_with_centers(otsu_mask, labelmap_page, stats_page, centers_page), "Components with centers")

    finalize_figure(fig, "page99_connected_components.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 99 - Connected components"))
