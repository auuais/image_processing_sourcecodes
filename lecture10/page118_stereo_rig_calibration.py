from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt

from common import StereoCalibration, calibrate_stereo, draw_text_block, finalize_figure, format_matrix, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 10 page 118: stereo rig calibration.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def build_rectified_preview(calibration: StereoCalibration) -> tuple[cv2.UMat | cv2.Mat | None, tuple[int, int, int, int], tuple[int, int, int, int]]:
    left = calibration.left_samples[0].image
    right = calibration.right_samples[0].image
    image_size = left.shape[1], left.shape[0]

    R1, R2, P1, P2, _Q, roi1, roi2 = cv2.stereoRectify(
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
        R1,
        P1,
        image_size,
        cv2.CV_16SC2,
    )
    right_map = cv2.initUndistortRectifyMap(
        calibration.right_camera_matrix,
        calibration.right_dist_coeffs,
        R2,
        P2,
        image_size,
        cv2.CV_16SC2,
    )
    rect_left = cv2.remap(left, left_map[0], left_map[1], cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, right_map[0], right_map[1], cv2.INTER_LINEAR)
    preview = cv2.hconcat([rect_left, rect_right])
    for y in range(40, preview.shape[0], 40):
        cv2.line(preview, (0, y), (preview.shape[1], y), (0, 255, 255), 1, cv2.LINE_AA)
    return preview, roi1, roi2


def main() -> None:
    args = parse_args()
    calibration = calibrate_stereo()
    rectified_preview, roi1, roi2 = build_rectified_preview(calibration)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_bgr(axes[0, 0], calibration.left_samples[0].visualized, "Left Camera Corners")
    plot_bgr(axes[0, 1], calibration.right_samples[0].visualized, "Right Camera Corners")
    plot_bgr(axes[1, 0], rectified_preview, "Rectified Stereo Preview")

    summary = (
        f"Stereo pairs: {len(calibration.object_points)}\n"
        f"Pattern size: {calibration.pattern_size}\n"
        f"RMS reprojection error: {calibration.rms:.6f}\n"
        f"Translation vector T:\n{format_matrix(calibration.translation)}\n\n"
        f"Left camera matrix K1:\n{format_matrix(calibration.left_camera_matrix)}\n\n"
        f"Right camera matrix K2:\n{format_matrix(calibration.right_camera_matrix)}\n\n"
        f"Rectified ROI left: {roi1}\n"
        f"Rectified ROI right: {roi2}"
    )
    draw_text_block(axes[1, 1], "Stereo Calibration Output", summary)
    finalize_figure(fig, "page118_stereo_rig_calibration.png", show=args.show)


if __name__ == "__main__":
    main()
