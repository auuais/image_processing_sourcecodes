from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import calibrate_pinhole, finalize_figure, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 10 page 119: distorting and undistorting points.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = calibrate_pinhole()
    sample = calibration.samples[0]

    undistorted_normalized = cv2.undistortPoints(sample.corners, calibration.camera_matrix, calibration.dist_coeffs)
    homogeneous = np.column_stack(
        [undistorted_normalized.squeeze(), np.ones((len(undistorted_normalized),), dtype=np.float32)]
    ).astype(np.float32)
    reprojected, _ = cv2.projectPoints(
        homogeneous,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        calibration.camera_matrix,
        None,
    )

    undistorted_vis = sample.image.copy()
    reprojection_vis = sample.image.copy()
    for point in sample.corners.squeeze().astype(np.float32):
        cv2.circle(undistorted_vis, tuple(np.round(point).astype(int)), 8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(reprojection_vis, tuple(np.round(point).astype(int)), 8, (0, 255, 0), 2, cv2.LINE_AA)
    for point in reprojected.squeeze().astype(np.float32):
        cv2.circle(reprojection_vis, tuple(np.round(point).astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_bgr(axes[0], sample.visualized, "Detected Distorted Corners")
    plot_bgr(axes[1], undistorted_vis, "Original Corner Locations")
    plot_bgr(axes[2], reprojection_vis, "Reprojected Undistorted Points")
    finalize_figure(fig, "page119_distorting_and_undistorting_points.png", show=args.show)


if __name__ == "__main__":
    main()
