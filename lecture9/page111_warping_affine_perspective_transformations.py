from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import CIRCLESGRID_PATH, finalize_figure, plot_bgr, read_color, save_image


AFFINE_DST = np.array([[0, 240], [0, 0], [240, 0]], dtype=np.float32)
PERSPECTIVE_DST = np.array([[0, 240], [0, 0], [240, 0], [240, 240]], dtype=np.float32)
AFFINE_SRC_DEFAULT = np.array([[41, 279], [39, 40], [280, 39]], dtype=np.float32)
PERSPECTIVE_SRC_DEFAULT = np.array([[39, 279], [40, 40], [279, 40], [280, 279]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 9 page 111: affine and perspective warping demo.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    parser.add_argument("--interactive", dest="interactive", action="store_true", help="Select source points manually with the mouse.")
    parser.add_argument("--non-interactive", dest="interactive", action="store_false", help="Use the built-in source points instead of mouse selection.")
    parser.set_defaults(interactive=True)
    return parser.parse_args()


def draw_points(image: np.ndarray, points: np.ndarray, closed: bool = False) -> np.ndarray:
    output = image.copy()
    for index, point in enumerate(points.astype(int), start=1):
        cv2.circle(output, tuple(point), 8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, str(index), tuple(point + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if closed and len(points) >= 4:
        cv2.polylines(output, [points.astype(int)], True, (0, 200, 255), 2, cv2.LINE_AA)
    return output


def select_points(image: np.ndarray, point_count: int) -> np.ndarray:
    selected: list[list[int]] = []
    canvas = image.copy()

    def mouse_callback(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONUP and len(selected) < point_count:
            selected.append([x, y])
            cv2.circle(canvas, (x, y), 8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(len(selected)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.namedWindow("select_points")
    cv2.setMouseCallback("select_points", mouse_callback)
    try:
        while len(selected) < point_count:
            cv2.imshow("select_points", canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
        if len(selected) != point_count:
            raise RuntimeError(f"Expected {point_count} points, received {len(selected)}.")
    finally:
        cv2.destroyWindow("select_points")

    return np.array(selected, dtype=np.float32)


def main() -> None:
    args = parse_args()
    image = read_color(CIRCLESGRID_PATH)

    affine_src = select_points(image, 3) if args.interactive else AFFINE_SRC_DEFAULT.copy()
    affine_vis = draw_points(image, affine_src)
    affine_matrix = cv2.getAffineTransform(affine_src, AFFINE_DST)
    affine_unwarped = cv2.warpAffine(image, affine_matrix, (240, 240))
    inverse_affine = cv2.invertAffineTransform(affine_matrix)
    affine_restored = cv2.warpAffine(affine_unwarped, inverse_affine, (image.shape[1], image.shape[0]))

    rotation_matrix = cv2.getRotationMatrix2D(tuple(affine_src[0]), 6.0, 1.0)
    affine_rotated = cv2.warpAffine(image, rotation_matrix, (240, 240))

    perspective_src = select_points(image, 4) if args.interactive else PERSPECTIVE_SRC_DEFAULT.copy()
    perspective_vis = draw_points(image, perspective_src, closed=True)
    perspective_matrix = cv2.getPerspectiveTransform(perspective_src, PERSPECTIVE_DST)
    perspective_unwarped = cv2.warpPerspective(image, perspective_matrix, (240, 240))

    save_image(CIRCLESGRID_PATH.parent.parent / "output" / "page111_affine_unwarped.png", affine_unwarped)
    save_image(CIRCLESGRID_PATH.parent.parent / "output" / "page111_affine_restored.png", affine_restored)
    save_image(CIRCLESGRID_PATH.parent.parent / "output" / "page111_affine_rotated.png", affine_rotated)
    save_image(CIRCLESGRID_PATH.parent.parent / "output" / "page111_perspective_unwarped.png", perspective_unwarped)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_bgr(axes[0, 0], affine_vis, "Affine Source Points")
    plot_bgr(axes[0, 1], affine_unwarped, "Affine Warp")
    plot_bgr(axes[0, 2], affine_restored, "Inverse Affine Warp")
    plot_bgr(axes[1, 0], perspective_vis, "Perspective Source Points")
    plot_bgr(axes[1, 1], perspective_unwarped, "Perspective Warp")
    plot_bgr(axes[1, 2], affine_rotated, "Rotation Warp")
    finalize_figure(fig, "page111_warping_affine_perspective_transformations.png", show=args.show)


if __name__ == "__main__":
    main()
