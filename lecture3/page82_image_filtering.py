from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_lena_color, parse_show_flag, plot_bgr


def main(show: bool = False) -> None:
    image = load_lena_color().astype(np.float32) / 255.0

    rng = np.random.default_rng(0)
    noise = 0.2 * rng.random(image.shape).astype(np.float32)
    noised = np.clip(image + noise, 0.0, 1.0)

    gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
    median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
    bilateral = cv2.bilateralFilter(noised, -1, 0.3, 10)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    plot_bgr(axes[0], image, "Original")
    plot_bgr(axes[1], noised, "Noised")
    plot_bgr(axes[2], gauss_blur, "Gaussian blur")
    plot_bgr(axes[3], median_blur, "Median blur")
    plot_bgr(axes[4], bilateral, "Bilateral filter")

    finalize_figure(fig, "page82_image_filtering.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 82 - Image filtering"))
