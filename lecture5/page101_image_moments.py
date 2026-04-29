from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, parse_show_flag, plot_bgr, plot_gray


def main(show: bool = False) -> None:
    image = np.zeros((480, 640), np.uint8)
    cv2.ellipse(image, (320, 240), (200, 100), 0, 0, 360, 255, -1)

    moments = cv2.moments(image)
    for name, value in moments.items():
        print(f"{name}\t{value}")

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    print(f"Center X estimated: {center_x:.2f}")
    print(f"Center Y estimated: {center_y:.2f}")

    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(display, (int(center_x), int(center_y)), 7, (0, 0, 255), -1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_gray(axes[0], image, "Ellipse")
    plot_bgr(axes[1], display, "Estimated center")

    finalize_figure(fig, "page101_image_moments.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 101 - Image moments"))
