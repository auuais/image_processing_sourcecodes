from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import choose_image_name, finalize_figure, load_color_image, plot_bgr


def kmeans_lab(image_bgr: np.ndarray, num_classes: int = 8) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    data = image_lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _compactness, labels, centers = cv2.kmeans(
        data,
        num_classes,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    segmented_lab = centers[labels.flatten()].reshape(image_lab.shape).astype(np.float32)
    segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2BGR)
    return (segmented * 255.0).clip(0, 255).astype(np.uint8)


def main(show: bool = False, image_name: str | None = None) -> None:
    selected_name = choose_image_name(image_name, color_only=True, prompt="Select image for page 58 K-means segmentation:")
    image = load_color_image(selected_name)
    segmented = kmeans_lab(image)
    image_tag = Path(selected_name).stem

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_bgr(axes[0], image, f"Original - {selected_name}")
    plot_bgr(axes[1], segmented, "Segmented")

    finalize_figure(fig, f"page58_kmeans_segmentation_{image_tag}.png", show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 58 - K-means segmentation")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--image", help="Optional image name from the data folder.")
    args = parser.parse_args()
    main(show=args.show, image_name=args.image)
