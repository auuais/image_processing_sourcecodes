from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import finalize_figure, load_gray_image, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("coins.png")
    otsu_thr, otsu_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f"Estimated threshold (Otsu): {otsu_thr:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_gray(axes[0], image, "Original")
    plot_gray(axes[1], otsu_mask, "Otsu threshold")

    finalize_figure(fig, "page97_otsu_thresholding.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 97 - Otsu thresholding"))
