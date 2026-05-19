from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt

from common import calibrate_pinhole, draw_text_block, finalize_figure, format_matrix, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 10 page 116: pinhole camera model calibration.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = calibrate_pinhole()

    sample = calibration.samples[0]
    undistorted = cv2.undistort(sample.image, calibration.camera_matrix, calibration.dist_coeffs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_bgr(axes[0, 0], sample.image, "Original Chessboard Image")
    plot_bgr(axes[0, 1], sample.visualized, "Detected Chessboard Corners")
    plot_bgr(axes[1, 0], undistorted, "Undistorted Preview")

    summary = (
        f"Detected samples: {len(calibration.samples)}\n"
        f"Pattern size: {calibration.pattern_size}\n"
        f"RMS reprojection error: {calibration.rms:.6f}\n\n"
        f"camera_mat.npy\n{format_matrix(calibration.camera_matrix)}\n\n"
        f"dist_coefs.npy\n{format_matrix(calibration.dist_coeffs)}"
    )
    draw_text_block(axes[1, 1], "Calibration Output", summary)
    finalize_figure(fig, "page116_pinhole_camera_model_calibration.png", show=args.show)


if __name__ == "__main__":
    main()
