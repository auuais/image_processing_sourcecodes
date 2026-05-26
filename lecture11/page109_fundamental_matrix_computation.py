from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import calibrate_stereo, draw_text_block, finalize_figure, format_matrix, plot_bgr, select_sample_by_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 11 page 109: fundamental matrix computation.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def draw_epilines(image: np.ndarray, lines: np.ndarray, points: np.ndarray, seed: int) -> np.ndarray:
    output = image.copy()
    rng = np.random.default_rng(seed)
    height, width = output.shape[:2]
    for line, point in zip(lines, points):
        color = tuple(int(v) for v in rng.integers(40, 255, size=3))
        a, b, c = line
        if abs(b) > 1e-6:
            x0, y0 = 0, int(round(-c / b))
            x1, y1 = width - 1, int(round(-(c + a * (width - 1)) / b))
        else:
            x = int(round(-c / a)) if abs(a) > 1e-6 else 0
            x0, y0 = x, 0
            x1, y1 = x, height - 1
        cv2.line(output, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(output, tuple(np.round(point).astype(int)), 5, color, -1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    calibration = calibrate_stereo()

    all_left_ud = cv2.undistortPoints(
        calibration.left_points.reshape(-1, 1, 2),
        calibration.left_camera_matrix,
        calibration.left_dist_coeffs,
        P=calibration.left_camera_matrix,
    )
    all_right_ud = cv2.undistortPoints(
        calibration.right_points.reshape(-1, 1, 2),
        calibration.right_camera_matrix,
        calibration.right_dist_coeffs,
        P=calibration.right_camera_matrix,
    )

    fundamental_est, mask = cv2.findFundamentalMat(all_left_ud, all_right_ud, cv2.FM_LMEDS)
    if fundamental_est is None or mask is None:
        raise RuntimeError("Fundamental matrix estimation failed.")
    essential_from_f = calibration.right_camera_matrix.T @ fundamental_est @ calibration.left_camera_matrix

    left_sample = select_sample_by_name(calibration.left_samples, "left14")
    right_sample = select_sample_by_name(calibration.right_samples, "right14")
    sample_index = next(
        index for index, sample in enumerate(calibration.left_samples) if sample.path.stem == left_sample.path.stem
    )
    left_points = cv2.undistortPoints(
        calibration.left_points[sample_index],
        calibration.left_camera_matrix,
        calibration.left_dist_coeffs,
        P=calibration.left_camera_matrix,
    ).reshape(-1, 2)
    right_points = cv2.undistortPoints(
        calibration.right_points[sample_index],
        calibration.right_camera_matrix,
        calibration.right_dist_coeffs,
        P=calibration.right_camera_matrix,
    ).reshape(-1, 2)

    subset_indices = np.linspace(0, len(left_points) - 1, 10, dtype=int)
    left_points_subset = left_points[subset_indices]
    right_points_subset = right_points[subset_indices]

    left_undistorted = cv2.undistort(left_sample.image, calibration.left_camera_matrix, calibration.left_dist_coeffs)
    right_undistorted = cv2.undistort(right_sample.image, calibration.right_camera_matrix, calibration.right_dist_coeffs)

    left_lines = cv2.computeCorrespondEpilines(right_points_subset.reshape(-1, 1, 2), 2, fundamental_est).reshape(-1, 3)
    right_lines = cv2.computeCorrespondEpilines(left_points_subset.reshape(-1, 1, 2), 1, fundamental_est).reshape(-1, 3)
    left_epi = draw_epilines(left_undistorted, left_lines, left_points_subset, seed=109)
    right_epi = draw_epilines(right_undistorted, right_lines, right_points_subset, seed=209)

    inlier_count = int(mask.ravel().sum())
    fundamental_reference = calibration.fundamental
    scale = float((fundamental_est * fundamental_reference).sum() / (fundamental_reference * fundamental_reference).sum())
    fundamental_delta = np.linalg.norm(fundamental_est - scale * fundamental_reference)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_bgr(axes[0, 0], left_epi, "Left Epipolar Lines")
    plot_bgr(axes[0, 1], right_epi, "Right Epipolar Lines")

    summary_f = (
        f"Inliers: {inlier_count}/{len(mask)}\n"
        f"Estimated F:\n{format_matrix(fundamental_est)}\n\n"
        f"Stereo-calibration F:\n{format_matrix(fundamental_reference)}\n\n"
        f"Scale-aligned delta: {fundamental_delta:.6f}"
    )
    draw_text_block(axes[1, 0], "Fundamental Matrix", summary_f)

    summary_e = (
        f"E = K_r^T F K_l\n\n"
        f"Estimated E:\n{format_matrix(essential_from_f)}\n\n"
        f"Stereo-calibration E:\n{format_matrix(calibration.essential)}"
    )
    draw_text_block(axes[1, 1], "Essential Matrix", summary_e)
    finalize_figure(fig, "page109_fundamental_matrix_computation.png", show=args.show)


if __name__ == "__main__":
    main()
