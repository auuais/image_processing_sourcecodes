from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("binary_blobs.png")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    erode_kernel = np.ones((3, 3), dtype=np.uint8)
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, erode_kernel, iterations=10)
    dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, erode_kernel, iterations=10)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ellipse_kernel, iterations=5)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ellipse_kernel, iterations=5)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, ellipse_kernel)

    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    entries = [
        ("Binary", binary),
        ("Erode 10 times", eroded),
        ("Dilate 10 times", dilated),
        ("Open 5 times", opened),
        ("Close 5 times", closed),
        ("Gradient", gradient),
    ]

    for ax, (title, result) in zip(axes.flat, entries):
        plot_gray(ax, result, title)

    finalize_figure(fig, "page71_morphological_filter.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 71 - Morphological filter"))
