from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, choose_image_name, finalize_figure, load_color_image, plot_bgr, save_image


RECT_COLOR = (0, 255, 0)
PAINT_COLORS = {
    cv2.GC_BGD: (0, 0, 255),
    cv2.GC_FGD: (255, 255, 255),
}


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    foreground_mask = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))[:, :, None]
    foreground = np.where(foreground_mask, image, 0)
    return foreground


def ratio_point(image: np.ndarray, x_ratio: float, y_ratio: float) -> tuple[int, int]:
    height, width = image.shape[:2]
    x = int(round(x_ratio * max(width - 1, 1)))
    y = int(round(y_ratio * max(height - 1, 1)))
    return x, y


def centered_rect(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x = int(round(width * 0.12))
    y = int(round(height * 0.10))
    w = max(20, int(round(width * 0.72)))
    h = max(20, int(round(height * 0.78)))
    w = min(w, width - x - 1 if width - x > 1 else width - x)
    h = min(h, height - y - 1 if height - y > 1 else height - y)
    return x, y, max(w, 1), max(h, 1)


def demo_grabcut(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.zeros(image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    rect = centered_rect(image)

    rect_view = image.copy()
    cv2.rectangle(rect_view, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), RECT_COLOR, 3)

    cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    initial = apply_mask(image, mask)

    scribble_view = rect_view.copy()
    radius = max(6, min(image.shape[:2]) // 50)
    fg_ratios = [(0.42, 0.28), (0.48, 0.46), (0.44, 0.64), (0.55, 0.80)]
    bg_ratios = [(0.04, 0.04), (0.96, 0.04), (0.96, 0.96), (0.88, 0.48)]

    for x_ratio, y_ratio in fg_ratios:
        point = ratio_point(image, x_ratio, y_ratio)
        cv2.circle(mask, point, 10, cv2.GC_FGD, -1)
        cv2.circle(scribble_view, point, radius, PAINT_COLORS[cv2.GC_FGD], -1)
    for x_ratio, y_ratio in bg_ratios:
        point = ratio_point(image, x_ratio, y_ratio)
        cv2.circle(mask, point, 10, cv2.GC_BGD, -1)
        cv2.circle(scribble_view, point, radius, PAINT_COLORS[cv2.GC_BGD], -1)

    cv2.grabCut(image, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    refined = apply_mask(image, mask)
    return rect_view, initial, refined


def print_interactive_help() -> None:
    print("GrabCut interactive controls:")
    print("  r : rectangle mode")
    print("  f : foreground brush")
    print("  b : background brush")
    print("  Left drag : draw rectangle or scribbles")
    print("  a : apply GrabCut")
    print("  s : save editor view and result")
    print("  c : clear and restart")
    print("  Esc : exit")
    print("Suggested order: draw rectangle -> press 'a' -> refine with f/b -> press 'a' again.")


def run_interactive(image: np.ndarray, image_name: str) -> None:
    window_editor = "grabcut editor"
    window_result = "grabcut result"
    annotated = image.copy()
    result = np.zeros_like(image)
    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    rect_mode = True
    rect = None
    start_point = (0, 0)
    mouse_pressed = False
    current_label = cv2.GC_FGD
    has_result = False
    image_tag = Path(image_name).stem

    print_interactive_help()

    def redraw_preview() -> np.ndarray:
        preview = annotated.copy()
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(preview, (x, y), (x + w, y + h), RECT_COLOR, 2)
        mode_name = "rectangle" if rect_mode else ("foreground" if current_label == cv2.GC_FGD else "background")
        cv2.putText(
            preview,
            f"Mode: {mode_name} | a apply | s save | c clear | esc exit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal rect, mouse_pressed, start_point
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pressed = True
            start_point = (x, y)
            if not rect_mode:
                cv2.circle(mask, (x, y), 5, current_label, -1)
                cv2.circle(annotated, (x, y), 5, PAINT_COLORS[current_label], -1)
        elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
            if rect_mode:
                x0, y0 = start_point
                rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            else:
                cv2.circle(mask, (x, y), 5, current_label, -1)
                cv2.circle(annotated, (x, y), 5, PAINT_COLORS[current_label], -1)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
            if rect_mode:
                x0, y0 = start_point
                rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            else:
                cv2.circle(mask, (x, y), 5, current_label, -1)
                cv2.circle(annotated, (x, y), 5, PAINT_COLORS[current_label], -1)

    cv2.namedWindow(window_editor, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_editor, mouse_callback)

    while True:
        cv2.imshow(window_editor, redraw_preview())
        cv2.imshow(window_result, result)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break
        if key == ord("r"):
            rect_mode = True
        elif key == ord("f"):
            rect_mode = False
            current_label = cv2.GC_FGD
        elif key == ord("b"):
            rect_mode = False
            current_label = cv2.GC_BGD
        elif key == ord("c"):
            annotated[:] = image
            result[:] = 0
            mask[:] = cv2.GC_PR_BGD
            bg_model[:] = 0
            fg_model[:] = 0
            rect = None
            has_result = False
        elif key == ord("a"):
            if rect is None:
                print("Draw a rectangle before applying GrabCut.")
                continue
            if not has_result:
                bg_model[:] = 0
                fg_model[:] = 0
                cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
                has_result = True
            else:
                cv2.grabCut(image, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
            result[:] = apply_mask(image, mask)
        elif key == ord("s"):
            save_image(OUTPUT_DIR / f"page62_grabcut_interactive_editor_{image_tag}.png", redraw_preview())
            save_image(OUTPUT_DIR / f"page62_grabcut_interactive_result_{image_tag}.png", result)

    cv2.destroyAllWindows()


def main(show: bool = False, interactive: bool = True, image_name: str | None = None) -> None:
    selected_name = choose_image_name(image_name, color_only=True, prompt="Select image for page 62 GrabCut segmentation:")
    image = load_color_image(selected_name)
    rect_view, initial, refined = demo_grabcut(image)
    image_tag = Path(selected_name).stem

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plot_bgr(axes[0], rect_view, f"Initial rectangle - {selected_name}")
    plot_bgr(axes[1], initial, "After rectangle GrabCut")
    plot_bgr(axes[2], refined, "Refined result")
    finalize_figure(fig, f"page62_grabcut_segmentation_{image_tag}.png", show)

    if interactive:
        run_interactive(image, selected_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 62 - GrabCut segmentation")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--image", help="Optional image name from the data folder.")
    parser.add_argument("--demo-only", action="store_true", help="Skip the interactive GrabCut editor.")
    args = parser.parse_args()
    main(show=args.show, interactive=not args.demo_only, image_name=args.image)
