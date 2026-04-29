from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, parse_show_flag, plot_bgr


def main(show: bool = False) -> None:
    rng = np.random.default_rng(1)

    ellipse_canvas = np.full((512, 512, 3), 255, np.uint8)
    axes = (180, 110)
    angle = 28
    center = (256, 256)
    ellipse_points = cv2.ellipse2Poly(center, axes, angle, 0, 360, 4).astype(np.int32)
    ellipse_points += rng.integers(-10, 11, size=ellipse_points.shape, dtype=np.int32)

    cv2.ellipse(ellipse_canvas, center, axes, angle, 0, 360, (0, 200, 0), 2)
    for point in ellipse_points:
        cv2.circle(ellipse_canvas, tuple(point), 3, (0, 0, 255), -1)

    fitted_ellipse = cv2.fitEllipse(ellipse_points.reshape(-1, 1, 2))
    cv2.ellipse(ellipse_canvas, fitted_ellipse, (0, 0, 0), 2)

    line_canvas = np.full((512, 512, 3), 255, np.uint8)
    xs = np.arange(20, 492, 8).reshape(-1, 1)
    ys = (0.75 * xs + 40).astype(np.int32)
    line_points = np.hstack([xs, ys]).astype(np.int32)
    line_points += rng.integers(-10, 11, size=line_points.shape, dtype=np.int32)

    cv2.line(line_canvas, (20, 55), (492, 410), (0, 200, 0), 2)
    for point in line_points:
        cv2.circle(line_canvas, tuple(point), 3, (0, 0, 255), -1)

    vx, vy, x, y = cv2.fitLine(line_points.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
    y0 = int((-x * vy / vx) + y)
    y1 = int(((512 - x) * vy / vx) + y)
    cv2.line(line_canvas, (0, y0), (512, y1), (0, 0, 0), 2)

    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6))
    plot_bgr(axes_plot[0], ellipse_canvas, "Fit ellipse")
    plot_bgr(axes_plot[1], line_canvas, "Fit line")

    finalize_figure(fig, "page100_fit_lines_and_circles.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 100 - Fitting lines and circles"))
