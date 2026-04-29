from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt

from common import finalize_figure, load_gray_image, normalize_channel, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("camera.png").astype("float32") / 255.0

    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, ktype=cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())

    filtered = cv2.filter2D(image, -1, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_gray(axes[0], image, "Image")
    plot_gray(axes[1], normalize_channel(kernel), "Gabor kernel")
    plot_gray(axes[2], filtered, "Filtered")

    finalize_figure(fig, "page66_gabor_filter.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 66 - Gabor filter"))
