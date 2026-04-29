from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    detect_fast_keypoints,
    finalize_figure,
    load_color_image,
    normalize_channel,
    parse_show_flag,
    plot_bgr,
    plot_gray,
    top_keypoints,
)


def main(show: bool = False) -> None:
    image = load_color_image("scene01.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    harris_response = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
    harris_response = cv2.dilate(harris_response, None)
    harris_overlay = image.copy()
    harris_overlay[harris_response > 0.1 * harris_response.max()] = (0, 0, 255)

    fast_nms = top_keypoints(detect_fast_keypoints(gray, threshold=30, nonmax_suppression=True), 250)
    fast_raw = top_keypoints(detect_fast_keypoints(gray, threshold=30, nonmax_suppression=False), 600)
    fast_nms_view = cv2.drawKeypoints(image, fast_nms, None, (0, 255, 0))
    fast_raw_view = cv2.drawKeypoints(image, fast_raw, None, (0, 255, 0))

    print(f"Harris thresholded pixels: {int(np.count_nonzero(harris_response > 0.1 * harris_response.max()))}")
    print(f"FAST corners with NMS: {len(fast_nms)}")
    print(f"FAST corners without NMS (top shown): {len(fast_raw)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_bgr(axes[0, 0], image, "Original scene")
    plot_bgr(axes[0, 1], harris_overlay, "Harris corners")
    plot_gray(axes[1, 0], normalize_channel(harris_response), "Harris response map")
    plot_bgr(axes[1, 1], fast_nms_view, "FAST corners with NMS")

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    plot_bgr(axes2[0], fast_nms_view, "FAST corners with NMS")
    plot_bgr(axes2[1], fast_raw_view, "FAST corners without NMS")

    finalize_figure(fig, "page70_harris_fast_corners_overview.png", show=False)
    finalize_figure(fig2, "page70_harris_fast_corner_detection.png", show=show)


if __name__ == "__main__":
    main(parse_show_flag("Page 70 - Harris and FAST corner detection"))
