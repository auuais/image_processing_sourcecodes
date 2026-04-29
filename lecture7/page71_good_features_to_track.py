from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import detect_gftt_points, finalize_figure, load_gray_image, parse_show_flag, plot_bgr, plot_gray


def main(show: bool = False) -> None:
    gray = load_gray_image("scene01.png")
    corners = detect_gftt_points(gray, max_corners=100, quality_level=0.05, min_distance=10)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)

    print(f"Good Features to Track corners: {len(corners)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_gray(axes[0], gray, "Original grayscale image")
    plot_bgr(axes[1], overlay, "Good Features to Track")
    finalize_figure(fig, "page71_good_features_to_track.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 71 - Good Features to Track"))
