from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import finalize_figure, load_gray_image, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("page.png")

    threshold_used, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    print(f"Threshold used: {threshold_used}")

    adaptive_mask = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        10,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_gray(axes[0], image, "Original")
    plot_gray(axes[1], mask, "Binary threshold")
    plot_gray(axes[2], adaptive_mask, "Adaptive threshold")

    finalize_figure(fig, "page69_image_thresholding.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 69 - Image thresholding"))
