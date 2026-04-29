from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_lena_color, load_lena_gray, parse_show_flag, plot_bgr, plot_gray


def main(show: bool = False) -> None:
    gray = load_lena_gray()
    gray_equalized = cv2.equalizeHist(gray)

    hist_gray, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_equalized, _ = np.histogram(gray_equalized, bins=256, range=(0, 256))

    color = load_lena_color()
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    color_equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plot_gray(axes[0, 0], gray, "Original gray")
    axes[0, 1].bar(range(256), hist_gray, width=1.0, color="gray")
    axes[0, 1].set_title("Gray histogram")
    axes[0, 1].set_xlim(0, 255)
    plot_gray(axes[0, 2], gray_equalized, "Equalized gray")

    axes[1, 0].bar(range(256), hist_equalized, width=1.0, color="black")
    axes[1, 0].set_title("Equalized histogram")
    axes[1, 0].set_xlim(0, 255)
    plot_bgr(axes[1, 1], color, "Original color")
    plot_bgr(axes[1, 2], color_equalized, "Equalized color (V channel)")

    finalize_figure(fig, "page81_histogram_equalization.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 81 - Histogram equalization"))
