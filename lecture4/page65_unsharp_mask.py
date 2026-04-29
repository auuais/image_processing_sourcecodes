from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import finalize_figure, load_color_image, normalize_channel, parse_show_flag, plot_bgr, plot_gray


def main(show: bool = False) -> None:
    image = load_color_image("astronaut.png")

    ksize = 11
    alpha = 2.0
    gaussian_1d = cv2.getGaussianKernel(ksize, 0)
    kernel = -alpha * (gaussian_1d @ gaussian_1d.T)
    kernel[ksize // 2, ksize // 2] += 1.0 + alpha

    filtered = cv2.filter2D(image, -1, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_bgr(axes[0], image, "Original image")
    plot_gray(axes[1], normalize_channel(kernel), "Unsharp kernel")
    plot_bgr(axes[2], filtered, "Sharpened image")

    finalize_figure(fig, "page65_unsharp_mask.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 65 - Unsharp mask"))
