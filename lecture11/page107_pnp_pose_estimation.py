from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import build_object_points, calibrate_pinhole, draw_text_block, finalize_figure, format_matrix, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 11 page 107: pose estimation with solvePnP.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = calibrate_pinhole()
    sample = calibration.samples[0]
    object_points = build_object_points(calibration.pattern_size)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        sample.corners,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed.")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, calibration.camera_matrix, calibration.dist_coeffs)
    reprojection_error = np.linalg.norm(projected.reshape(-1, 2) - sample.corners.reshape(-1, 2), axis=1)

    pose_overlay = sample.image.copy()
    cv2.drawFrameAxes(pose_overlay, calibration.camera_matrix, calibration.dist_coeffs, rvec, tvec, 3.0, 3)
    for point in projected.reshape(-1, 2):
        cv2.circle(pose_overlay, tuple(np.round(point).astype(int)), 4, (0, 255, 0), -1, cv2.LINE_AA)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_bgr(axes[0, 0], sample.image, "Original Calibration Image")
    plot_bgr(axes[0, 1], sample.visualized, "Detected Chessboard Corners")
    plot_bgr(axes[1, 0], pose_overlay, "PnP Pose and Reprojected Points")

    summary = (
        f"Pattern size: {calibration.pattern_size}\n"
        f"PnP success: {success}\n"
        f"Mean reprojection error: {reprojection_error.mean():.6f}\n"
        f"Max reprojection error: {reprojection_error.max():.6f}\n\n"
        f"Rotation vector:\n{format_matrix(rvec)}\n\n"
        f"Translation vector:\n{format_matrix(tvec)}\n\n"
        f"Camera matrix:\n{format_matrix(calibration.camera_matrix)}"
    )
    draw_text_block(axes[1, 1], "PnP Summary", summary)
    finalize_figure(fig, "page107_pnp_pose_estimation.png", show=args.show)


if __name__ == "__main__":
    main()

