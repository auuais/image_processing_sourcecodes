from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import calibrate_stereo, draw_text_block, finalize_figure, plot_bgr, select_sample_by_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 11 page 108: stereo rectification.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def draw_horizontal_guides(image: np.ndarray, spacing: int = 40) -> np.ndarray:
    output = image.copy()
    for y in range(spacing, output.shape[0], spacing):
        cv2.line(output, (0, y), (output.shape[1], y), (0, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    calibration = calibrate_stereo()
    left_sample = select_sample_by_name(calibration.left_samples, "left14")
    right_sample = select_sample_by_name(calibration.right_samples, "right14")

    image_size = left_sample.image.shape[1], left_sample.image.shape[0]
    r1, r2, p1, p2, _q, roi1, roi2 = cv2.stereoRectify(
        calibration.left_camera_matrix,
        calibration.left_dist_coeffs,
        calibration.right_camera_matrix,
        calibration.right_dist_coeffs,
        image_size,
        calibration.rotation,
        calibration.translation,
    )

    left_map = cv2.initUndistortRectifyMap(
        calibration.left_camera_matrix,
        calibration.left_dist_coeffs,
        r1,
        p1,
        image_size,
        cv2.CV_16SC2,
    )
    right_map = cv2.initUndistortRectifyMap(
        calibration.right_camera_matrix,
        calibration.right_dist_coeffs,
        r2,
        p2,
        image_size,
        cv2.CV_16SC2,
    )
    left_rectified = cv2.remap(left_sample.image, left_map[0], left_map[1], cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_sample.image, right_map[0], right_map[1], cv2.INTER_LINEAR)

    original_pair = draw_horizontal_guides(cv2.hconcat([left_sample.image, right_sample.image]))
    rectified_pair = draw_horizontal_guides(cv2.hconcat([left_rectified, right_rectified]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_bgr(axes[0, 0], left_sample.image, "Left Original")
    plot_bgr(axes[0, 1], right_sample.image, "Right Original")
    plot_bgr(axes[1, 0], original_pair, "Original Pair with Scanlines")
    plot_bgr(axes[1, 1], rectified_pair, "Rectified Pair with Scanlines")

    summary = (
        f"Image size: {image_size}\n"
        f"Rectified ROI left: {roi1}\n"
        f"Rectified ROI right: {roi2}\n\n"
        f"R1 shape: {r1.shape}\n"
        f"R2 shape: {r2.shape}\n"
        f"P1 shape: {p1.shape}\n"
        f"P2 shape: {p2.shape}\n\n"
        "After rectification the epipolar lines become horizontal,\n"
        "which makes left-right correspondence search much simpler."
    )
    fig.text(0.02, 0.02, summary, family="monospace", fontsize=10, va="bottom")
    finalize_figure(fig, "page108_stereo_rectification.png", show=args.show)


if __name__ == "__main__":
    main()

