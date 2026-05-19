from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import calibrate_fisheye, draw_text_block, finalize_figure, format_matrix, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 10 page 117: fisheye camera model calibration.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = calibrate_fisheye()

    sample = calibration.samples[0]
    height, width = sample.image.shape[:2]
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        calibration.camera_matrix,
        calibration.dist_coeffs,
        (width, height),
        np.eye(3),
        balance=0.0,
    )
    undistorted = cv2.fisheye.undistortImage(
        sample.image,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        Knew=new_camera_matrix,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_bgr(axes[0, 0], sample.image, "Original Fisheye Image")
    plot_bgr(axes[0, 1], sample.visualized, "Detected Chessboard Corners")
    plot_bgr(axes[1, 0], undistorted, "Undistorted Fisheye Preview")

    summary = (
        f"Detected samples: {len(calibration.samples)}\n"
        f"Pattern size: {calibration.pattern_size}\n"
        f"RMS reprojection error: {calibration.rms:.6f}\n\n"
        f"camera_mat.npy\n{format_matrix(calibration.camera_matrix)}\n\n"
        f"dist_coefs.npy\n{format_matrix(calibration.dist_coeffs)}"
    )
    draw_text_block(axes[1, 1], "Calibration Output", summary)
    finalize_figure(fig, "page117_fisheye_camera_model_calibration.png", show=args.show)


if __name__ == "__main__":
    main()
