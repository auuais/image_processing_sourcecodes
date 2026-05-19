from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    MonoCalibration,
    build_object_points,
    calibrate_pinhole,
    draw_text_block,
    finalize_figure,
    format_matrix,
    plot_bgr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 10 pages 110-114: step-by-step camera calibration demonstration.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def make_montage(images: list[np.ndarray], columns: int = 4, cell_size: tuple[int, int] = (220, 165)) -> np.ndarray:
    resized = [cv2.resize(image, cell_size, interpolation=cv2.INTER_AREA) for image in images]
    rows = int(np.ceil(len(resized) / columns))
    blank = np.full((cell_size[1], cell_size[0], 3), 245, dtype=np.uint8)
    canvas_rows = []
    for row_index in range(rows):
        row_images = resized[row_index * columns : (row_index + 1) * columns]
        while len(row_images) < columns:
            row_images.append(blank.copy())
        canvas_rows.append(cv2.hconcat(row_images))
    return cv2.vconcat(canvas_rows)


def draw_extreme_corners(calibration: MonoCalibration) -> np.ndarray:
    sample = calibration.samples[0]
    width, height = calibration.pattern_size
    indices = [0, width - 1, width * height - 1, width * (height - 1)]
    labels = ["Click #1 (origin)", "Click #2", "Click #3", "Click #4"]
    output = sample.image.copy()
    for label, index in zip(labels, indices):
        point = sample.corners[index, 0].astype(int)
        cv2.circle(output, tuple(point), 11, (0, 0, 255), 3, cv2.LINE_AA)
        text_anchor = tuple(point + np.array([12, -12]))
        cv2.putText(output, label, text_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def draw_reprojection_overlay(calibration: MonoCalibration) -> np.ndarray:
    sample = calibration.samples[0]
    object_points = build_object_points(calibration.pattern_size)
    projected, _ = cv2.projectPoints(
        object_points,
        calibration.rvecs[0],
        calibration.tvecs[0],
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    overlay = sample.image.copy()
    for point in sample.corners.reshape(-1, 2):
        cv2.circle(overlay, tuple(np.round(point).astype(int)), 5, (0, 0, 255), 1, cv2.LINE_AA)
    for point in projected.reshape(-1, 2):
        cv2.circle(overlay, tuple(np.round(point).astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(overlay, "red: detected corners  yellow: reprojected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 255, 40), 2, cv2.LINE_AA)
    return overlay


def plot_extrinsics(ax, calibration: MonoCalibration) -> None:
    board_width = calibration.pattern_size[0] - 1
    board_height = calibration.pattern_size[1] - 1
    board_outline = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width, 0.0, 0.0],
            [board_width, board_height, 0.0],
            [0.0, board_height, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    ax.scatter([0], [0], [0], c="black", s=45)
    ax.text(0.0, 0.0, 0.0, "camera", color="black")
    ax.quiver(0, 0, 0, 4, 0, 0, color="red", linewidth=2)
    ax.quiver(0, 0, 0, 0, 4, 0, color="green", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 4, color="blue", linewidth=2)

    colors = plt.cm.tab20(np.linspace(0, 1, len(calibration.samples)))
    for sample_index, (rvec, tvec, color) in enumerate(zip(calibration.rvecs, calibration.tvecs, colors), start=1):
        rotation, _ = cv2.Rodrigues(rvec)
        transformed = (rotation @ board_outline.T).T + tvec.reshape(1, 3)
        ax.plot(transformed[:, 0], transformed[:, 1], transformed[:, 2], color=color, linewidth=1.5)
        center = transformed[:-1].mean(axis=0)
        ax.text(center[0], center[1], center[2], str(sample_index), color=color)

    ax.set_title("Extrinsic Parameters")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-58)


def main() -> None:
    args = parse_args()
    calibration = calibrate_pinhole()

    montage = make_montage([sample.image for sample in calibration.samples])
    clicked = draw_extreme_corners(calibration)
    extracted = calibration.samples[0].visualized
    reprojection = draw_reprojection_overlay(calibration)

    fig = plt.figure(figsize=(17, 13))
    grid = fig.add_gridspec(3, 2)

    plot_bgr(fig.add_subplot(grid[0, 0]), montage, "Calibration Images")
    plot_bgr(fig.add_subplot(grid[0, 1]), clicked, "Extreme Corners of the Pattern")
    plot_bgr(fig.add_subplot(grid[1, 0]), extracted, "Extracted Corners")
    plot_bgr(fig.add_subplot(grid[1, 1]), reprojection, "Image Points and Reprojected Grid")
    plot_extrinsics(fig.add_subplot(grid[2, 0], projection="3d"), calibration)

    summary = (
        f"Detected samples: {len(calibration.samples)}\n"
        f"Pattern size: {calibration.pattern_size}\n"
        f"RMS reprojection error: {calibration.rms:.4f}\n\n"
        f"Camera matrix:\n{format_matrix(calibration.camera_matrix)}\n\n"
        f"Distortion coefficients:\n{format_matrix(calibration.dist_coeffs)}"
    )
    draw_text_block(fig.add_subplot(grid[2, 1]), "Calibration Summary", summary)

    finalize_figure(fig, "page110_step_by_step_demonstration.png", show=args.show)


if __name__ == "__main__":
    main()
