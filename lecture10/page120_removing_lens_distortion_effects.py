from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt

from common import calibrate_pinhole, finalize_figure, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 10 page 120: removing lens distortion effects.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = calibrate_pinhole()
    sample = calibration.samples[0]

    direct_undistort = cv2.undistort(sample.image, calibration.camera_matrix, calibration.dist_coeffs)
    height, width = sample.image.shape[:2]
    optimal_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.dist_coeffs,
        (width, height),
        0,
    )
    optimal_undistort = cv2.undistort(
        sample.image,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        None,
        optimal_camera_matrix,
    )

    roi_vis = optimal_undistort.copy()
    x, y, roi_w, roi_h = valid_roi
    if roi_w > 0 and roi_h > 0:
        cv2.rectangle(roi_vis, (x, y), (x + roi_w, y + roi_h), (0, 255, 0), 2, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_bgr(axes[0], sample.image, "Original Image")
    plot_bgr(axes[1], direct_undistort, "cv2.undistort")
    plot_bgr(axes[2], roi_vis, "Optimal New Camera Matrix")
    finalize_figure(fig, "page120_removing_lens_distortion_effects.png", show=args.show)


if __name__ == "__main__":
    main()
