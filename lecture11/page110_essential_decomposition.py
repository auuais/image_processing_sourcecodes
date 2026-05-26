from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import calibrate_stereo, draw_text_block, finalize_figure, format_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 11 page 110: essential decomposition into rotation and translation.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def plot_rotation_axes(ax, rotation: np.ndarray, title: str, color_scale: float = 1.0) -> None:
    origin = np.zeros(3)
    axes = rotation @ np.eye(3)
    colors = ["red", "green", "blue"]
    labels = ["x", "y", "z"]
    for axis_vector, color, label in zip(axes.T, colors, labels):
        ax.quiver(*origin, *(axis_vector * color_scale), color=color, linewidth=2)
        endpoint = axis_vector * color_scale
        ax.text(endpoint[0], endpoint[1], endpoint[2], label, color=color)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)


def main() -> None:
    args = parse_args()
    calibration = calibrate_stereo()
    rotation_1, rotation_2, translation = cv2.decomposeEssentialMat(calibration.essential)

    translation_unit = translation.reshape(-1) / np.linalg.norm(translation)
    reference_unit = calibration.translation.reshape(-1) / np.linalg.norm(calibration.translation)

    fig = plt.figure(figsize=(15, 10))
    plot_rotation_axes(fig.add_subplot(2, 2, 1, projection="3d"), rotation_1, "Candidate Rotation 1")
    plot_rotation_axes(fig.add_subplot(2, 2, 2, projection="3d"), rotation_2, "Candidate Rotation 2")
    plot_rotation_axes(fig.add_subplot(2, 2, 3, projection="3d"), calibration.rotation, "Stereo Calibration Rotation")

    summary = (
        f"Candidate rotation R1:\n{format_matrix(rotation_1)}\n\n"
        f"Candidate rotation R2:\n{format_matrix(rotation_2)}\n\n"
        f"Translation direction from E:\n{format_matrix(translation_unit.reshape(3, 1))}\n\n"
        f"Reference translation direction:\n{format_matrix(reference_unit.reshape(3, 1))}\n\n"
        "The decomposition returns translation only up to scale\n"
        "and sign. The physically valid solution is chosen later\n"
        "through cheirality constraints."
    )
    draw_text_block(fig.add_subplot(2, 2, 4), "Decomposition Summary", summary)
    finalize_figure(fig, "page110_essential_decomposition.png", show=args.show)


if __name__ == "__main__":
    main()
