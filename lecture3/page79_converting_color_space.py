from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_lena_color, parse_show_flag, plot_bgr, plot_gray


def main(show: bool = False) -> None:
    image = load_lena_color().astype(np.float32) / 255.0
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    boosted_hsv = hsv.copy()
    boosted_hsv[:, :, 2] = np.clip(boosted_hsv[:, :, 2] * 2.0, 0.0, 1.0)
    from_hsv = cv2.cvtColor(boosted_hsv, cv2.COLOR_HSV2BGR)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_bgr(axes[0], image, "Original image")
    plot_gray(axes[1], (gray * 255).astype(np.uint8), "Converted to grayscale")
    plot_bgr(axes[2], from_hsv, "V channel x 2 in HSV")

    finalize_figure(fig, "page79_converting_color_space.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 79 - Converting color space"))
