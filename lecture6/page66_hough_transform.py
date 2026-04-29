from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, parse_show_flag, plot_bgr, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("line_circle.png")

    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=200,
        param2=18,
        minRadius=20,
        maxRadius=70,
    )
    lines = cv2.HoughLinesP(
        image,
        1,
        np.pi / 180,
        80,
        minLineLength=80,
        maxLineGap=20,
    )

    debug = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            print(f"Detected line: ({x1}, {y1}) ({x2}, {y2})")
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            print(f"Detected circle: center=({x}, {y}), radius={r}")
            cv2.circle(debug, (x, y), r, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_gray(axes[0], image, "Original")
    plot_bgr(axes[1], debug, "Detected primitives")

    finalize_figure(fig, "page66_hough_transform.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 66 - Hough transform"))
