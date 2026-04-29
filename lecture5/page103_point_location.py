from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, plot_bgr


def selected_contour(image: np.ndarray) -> np.ndarray:
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def draw_point_test(image: np.ndarray, contour: np.ndarray, point: tuple[int, int], measure: bool = True) -> np.ndarray:
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color, [contour], -1, (0, 255, 0), 2)

    distance = cv2.pointPolygonTest(contour, point, measure)
    if distance > 0:
        point_color = (0, 255, 0)
    elif distance < 0:
        point_color = (0, 0, 255)
    else:
        point_color = (128, 0, 128)

    cv2.circle(color, point, 6, point_color, -1)
    cv2.putText(color, f"{distance:.2f}", (10, color.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return color


def run_interactive(image: np.ndarray, contour: np.ndarray) -> None:
    window_name = "Point location"
    state = {"measure": True, "image": draw_point_test(image, contour, (170, 180), True)}

    def callback(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            state["image"] = draw_point_test(image, contour, (x, y), state["measure"])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, callback)

    while True:
        cv2.imshow(window_name, state["image"])
        key = cv2.waitKey(30)
        if key == ord("m"):
            state["measure"] = not state["measure"]
        elif key == 27:
            break

    cv2.destroyWindow(window_name)


def main(show: bool = False, interactive: bool = False) -> None:
    image = load_gray_image("bnw_shapes.png")
    contour = selected_contour(image)

    sample_views = [
        ("Inside point", draw_point_test(image, contour, (170, 180))),
        ("On contour", draw_point_test(image, contour, (265, 180))),
        ("Outside point", draw_point_test(image, contour, (60, 60))),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, view) in zip(axes, sample_views):
        plot_bgr(ax, view, title)

    finalize_figure(fig, "page103_point_location.png", show)

    if interactive:
        run_interactive(image, contour)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 103 - Checking the location of points")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    main(show=args.show, interactive=args.interactive)
