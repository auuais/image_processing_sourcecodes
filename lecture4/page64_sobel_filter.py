from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import finalize_figure, load_gray_image, normalize_channel, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("camera.png")

    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_gray(axes[0], image, "Image")
    plot_gray(axes[1], normalize_channel(abs(dx)), "dI/dx")
    plot_gray(axes[2], normalize_channel(abs(dy)), "dI/dy")

    finalize_figure(fig, "page64_sobel_filter.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 64 - Sobel filter"))
