from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    circle_image = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(circle_image, (250, 250), 100, 255, -1)

    rect_image = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(rect_image, (100, 100), (400, 250), 255, -1)

    circle_and_rect = circle_image & rect_image
    circle_or_rect = circle_image | rect_image

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plot_gray(axes[0, 0], circle_image, "Circle")
    plot_gray(axes[0, 1], rect_image, "Rectangle")
    plot_gray(axes[1, 0], circle_and_rect, "Circle & rectangle")
    plot_gray(axes[1, 1], circle_or_rect, "Circle | rectangle")

    finalize_figure(fig, "page70_binary_operation.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 70 - Binary operation"))
