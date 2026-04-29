from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, choose_image_name, finalize_figure, load_color_image, plot_bgr, save_image


MARKER_COLORS = {
    1: (40, 220, 40),
    2: (220, 40, 40),
    3: (40, 40, 220),
    4: (220, 220, 40),
}


def scale_point(image: np.ndarray, x_ratio: float, y_ratio: float) -> tuple[int, int]:
    height, width = image.shape[:2]
    x = int(round(x_ratio * max(width - 1, 1)))
    y = int(round(y_ratio * max(height - 1, 1)))
    return x, y


def apply_predefined_markers(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    markers = np.zeros(image.shape[:2], dtype=np.int32)
    overlay = image.copy()

    seed_groups = {
        1: [(0.50, 0.22), (0.48, 0.35), (0.45, 0.53)],
        2: [(0.40, 0.38), (0.42, 0.55), (0.35, 0.68)],
        3: [(0.67, 0.22), (0.82, 0.39), (0.82, 0.68)],
        4: [(0.12, 0.12), (0.14, 0.82), (0.88, 0.88)],
    }

    radius = max(4, min(image.shape[:2]) // 70)
    for label, ratios in seed_groups.items():
        for x_ratio, y_ratio in ratios:
            point = scale_point(image, x_ratio, y_ratio)
            cv2.circle(markers, point, radius, label, -1)
            cv2.circle(overlay, point, radius, MARKER_COLORS[label], -1)

    return markers, overlay


def render_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    markers_copy = markers.copy()
    cv2.watershed(image, markers_copy)
    segmentation = np.zeros_like(image)

    for label, color in MARKER_COLORS.items():
        segmentation[markers_copy == label] = color

    segmentation[markers_copy == -1] = (255, 255, 255)
    return segmentation


def print_interactive_help() -> None:
    print("Watershed interactive controls:")
    print("  1-4 : choose seed label")
    print("  Left drag : paint seeds")
    print("  a : apply watershed")
    print("  s : save current seed image and result")
    print("  c : clear all seeds")
    print("  Esc : exit")


def run_interactive(image: np.ndarray, image_name: str) -> None:
    window_image = "image"
    window_seg = "segmentation"
    markers = np.zeros(image.shape[:2], dtype=np.int32)
    overlay = image.copy()
    segmentation = np.zeros_like(image)
    current_seed = 1
    mouse_pressed = False
    image_tag = Path(image_name).stem

    print_interactive_help()

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal mouse_pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pressed = True
            cv2.circle(markers, (x, y), 5, current_seed, cv2.FILLED)
            cv2.circle(overlay, (x, y), 5, MARKER_COLORS[current_seed], cv2.FILLED)
        elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
            cv2.circle(markers, (x, y), 5, current_seed, cv2.FILLED)
            cv2.circle(overlay, (x, y), 5, MARKER_COLORS[current_seed], cv2.FILLED)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False

    cv2.namedWindow(window_image, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_seg, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_image, mouse_callback)

    while True:
        preview = overlay.copy()
        cv2.putText(
            preview,
            f"Seed {current_seed} | 1-4 select | a apply | s save | c clear | esc exit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_image, preview)
        cv2.imshow(window_seg, segmentation)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break
        if key == ord("c"):
            markers[:] = 0
            overlay[:] = image
            segmentation[:] = 0
            continue
        if key >= ord("1") and key <= ord("4") and not mouse_pressed:
            current_seed = int(chr(key))
        if key == ord("a") and not mouse_pressed and np.any(markers > 0):
            segmentation[:] = render_watershed(image, markers)
        if key == ord("s"):
            save_image(OUTPUT_DIR / f"page60_watershed_interactive_seeds_{image_tag}.png", overlay)
            save_image(OUTPUT_DIR / f"page60_watershed_interactive_result_{image_tag}.png", segmentation)

    cv2.destroyAllWindows()


def main(show: bool = False, interactive: bool = True, image_name: str | None = None) -> None:
    selected_name = choose_image_name(image_name, color_only=True, prompt="Select image for page 60 Watershed segmentation:")
    image = load_color_image(selected_name)
    markers, overlay = apply_predefined_markers(image)
    segmentation = render_watershed(image, markers)
    image_tag = Path(selected_name).stem

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_bgr(axes[0], overlay, f"Image with seeds - {selected_name}")
    plot_bgr(axes[1], segmentation, "Watershed segmentation")
    finalize_figure(fig, f"page60_watershed_segmentation_{image_tag}.png", show)

    if interactive:
        run_interactive(image, selected_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 60 - Watershed segmentation")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--image", help="Optional image name from the data folder.")
    parser.add_argument("--demo-only", action="store_true", help="Skip the interactive seed editor.")
    args = parser.parse_args()
    main(show=args.show, interactive=not args.demo_only, image_name=args.image)
