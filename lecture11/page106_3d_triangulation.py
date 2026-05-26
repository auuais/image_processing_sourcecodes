from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import draw_text_block, finalize_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 11 page 106: 3D triangulation.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(11)

    p1 = np.eye(3, 4, dtype=np.float32)
    p2 = np.eye(3, 4, dtype=np.float32)
    p2[0, 3] = -1.0

    point_count = 8
    points3d = np.empty((4, point_count), dtype=np.float32)
    points3d[:3, :] = rng.normal(size=(3, point_count)).astype(np.float32)
    points3d[2, :] += 5.0
    points3d[3, :] = 1.0

    points1 = p1 @ points3d
    points1 = points1[:2, :] / points1[2, :]
    points1 += rng.normal(scale=1e-2, size=points1.shape).astype(np.float32)

    points2 = p2 @ points3d
    points2 = points2[:2, :] / points2[2, :]
    points2 += rng.normal(scale=1e-2, size=points2.shape).astype(np.float32)

    reconstructed = cv2.triangulatePoints(p1, p2, points1, points2)
    reconstructed /= reconstructed[3, :]

    original_xyz = points3d[:3, :].T
    reconstructed_xyz = reconstructed[:3, :].T
    reconstruction_error = np.linalg.norm(original_xyz - reconstructed_xyz, axis=1)

    fig = plt.figure(figsize=(15, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2], c="tab:blue", s=55, label="original")
    ax3d.scatter(reconstructed_xyz[:, 0], reconstructed_xyz[:, 1], reconstructed_xyz[:, 2], c="tab:orange", marker="^", s=45, label="reconstructed")
    for index, point in enumerate(original_xyz):
        ax3d.text(point[0], point[1], point[2], str(index), color="tab:blue")
    ax3d.set_title("3D Points")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left")

    ax_left = fig.add_subplot(2, 2, 2)
    ax_left.scatter(points1[0], points1[1], c="tab:red")
    ax_left.set_title("Left Image Points")
    ax_left.invert_yaxis()
    ax_left.set_aspect("equal")
    ax_left.grid(True, alpha=0.3)

    ax_right = fig.add_subplot(2, 2, 3)
    ax_right.scatter(points2[0], points2[1], c="tab:green")
    ax_right.set_title("Right Image Points")
    ax_right.invert_yaxis()
    ax_right.set_aspect("equal")
    ax_right.grid(True, alpha=0.3)

    summary = (
        f"P1:\n{p1}\n\n"
        f"P2:\n{p2}\n\n"
        f"Mean reconstruction error: {reconstruction_error.mean():.6f}\n"
        f"Max reconstruction error: {reconstruction_error.max():.6f}\n\n"
        f"Original points:\n{np.array2string(original_xyz, precision=4, suppress_small=True)}\n\n"
        f"Reconstructed points:\n{np.array2string(reconstructed_xyz, precision=4, suppress_small=True)}"
    )
    draw_text_block(fig.add_subplot(2, 2, 4), "Triangulation Summary", summary)
    finalize_figure(fig, "page106_3d_triangulation.png", show=args.show)


if __name__ == "__main__":
    main()
