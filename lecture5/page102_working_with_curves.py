from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, parse_show_flag, plot_bgr


def select_main_contour(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour, contours


def describe_curve(contour: np.ndarray) -> None:
    print(f"Area of contour is {cv2.contourArea(contour):.2f}")
    print(f"Signed area of contour is {cv2.contourArea(contour, True):.2f}")
    print(f"Length of closed contour is {cv2.arcLength(contour, True):.2f}")
    print(f"Length of open contour is {cv2.arcLength(contour, False):.2f}")
    hull = cv2.convexHull(contour)
    print(f"Convex status of contour is {cv2.isContourConvex(contour)}")
    print(f"Convex status of its hull is {cv2.isContourConvex(hull)}")


def build_views(image: np.ndarray, contour: np.ndarray, contours: list[np.ndarray], epsilon_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contour_view = color.copy()
    hull_view = color.copy()
    approx_view = color.copy()

    cv2.drawContours(contour_view, contours, -1, (0, 255, 0), 2)
    hull = cv2.convexHull(contour)
    cv2.drawContours(hull_view, [hull], -1, (0, 0, 255), 2)

    epsilon = epsilon_scale * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(approx_view, [approx], -1, (255, 0, 255), 2)

    return contour_view, hull_view, approx_view


def run_interactive(contour: np.ndarray, base_image: np.ndarray) -> None:
    window_name = "Curve approximation"

    def update(value: int) -> None:
        epsilon_scale = max(value, 1) * 0.1 / 255.0
        _, _, approx_view = build_views(base_image, contour, [contour], epsilon_scale)
        cv2.imshow(window_name, approx_view)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Epsilon", window_name, 25, 255, update)
    update(25)

    while True:
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyWindow(window_name)


def main(show: bool = False, interactive: bool = False) -> None:
    image = load_gray_image("bnw_shapes.png")
    contour, contours = select_main_contour(image)
    describe_curve(contour)

    contour_view, hull_view, approx_view = build_views(image, contour, contours, epsilon_scale=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_bgr(axes[0], contour_view, "Contours")
    plot_bgr(axes[1], hull_view, "Convex hull")
    plot_bgr(axes[2], approx_view, "Approx poly")

    finalize_figure(fig, "page102_working_with_curves.png", show)

    if interactive:
        run_interactive(contour, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 102 - Working with curves")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    main(show=args.show, interactive=args.interactive)
