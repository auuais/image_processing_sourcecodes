from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, choose_image_name, finalize_figure, load_color_image, plot_bgr, save_image


def kmeans_segmentation(image_bgr: np.ndarray, num_classes: int = 8, include_xy: bool = False, spatial_scale: float = 0.35) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    height, width = image.shape[:2]
    colors = image.reshape((-1, 3))

    if include_xy:
        xs = np.tile(np.arange(width, dtype=np.float32), height).reshape((-1, 1))
        ys = np.repeat(np.arange(height, dtype=np.float32), width).reshape((-1, 1))
        xs = (xs / max(width - 1, 1)) * 255.0 * spatial_scale
        ys = (ys / max(height - 1, 1)) * 255.0 * spatial_scale
        data = np.hstack([colors, xs, ys]).astype(np.float32)
    else:
        data = colors.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _compactness, labels, centers = cv2.kmeans(
        data,
        num_classes,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    segmented = centers[labels.flatten(), :3].reshape(image.shape)
    return np.clip(segmented, 0, 255).astype(np.uint8)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(
        ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))[:, :, None],
        image,
        0,
    )


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

    initial_view = image.copy()
    cv2.rectangle(initial_view, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)

    cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    rect_result = apply_mask(image, mask)

    refined_mask = mask.copy()
    refined_view = image.copy()
    fg_ratios = [(0.42, 0.28), (0.42, 0.46), (0.45, 0.64), (0.52, 0.80)]
    bg_ratios = [(0.04, 0.04), (0.96, 0.04), (0.96, 0.96), (0.88, 0.48)]
    radius = max(6, min(image.shape[:2]) // 50)

    for x_ratio, y_ratio in fg_ratios:
        point = ratio_point(image, x_ratio, y_ratio)
        cv2.circle(refined_mask, point, 10, cv2.GC_FGD, -1)
        cv2.circle(refined_view, point, radius, (255, 255, 255), -1)
    for x_ratio, y_ratio in bg_ratios:
        point = ratio_point(image, x_ratio, y_ratio)
        cv2.circle(refined_mask, point, 10, cv2.GC_BGD, -1)
        cv2.circle(refined_view, point, radius, (0, 0, 255), -1)

    cv2.grabCut(image, refined_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    refined_result = apply_mask(image, refined_mask)
    return initial_view, rect_result, refined_result


def print_grabcut_help() -> None:
    print("Assignment GrabCut controls:")
    print("  r : rectangle mode")
    print("  f : foreground brush")
    print("  b : background brush")
    print("  Left drag : draw rectangle or scribbles")
    print("  a : apply GrabCut")
    print("  s : save current editor view and result")
    print("  c : clear and restart")
    print("  Esc : exit")
    print("Suggested order: draw rectangle -> press 'a' -> refine with f/b -> press 'a' again.")


def run_interactive_grabcut(image: np.ndarray, output_dir: Path, image_name: str) -> None:
    window_editor = "assignment_grabcut_editor"
    window_result = "assignment_grabcut_result"
    annotated = image.copy()
    result = np.zeros_like(image)
    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    rect = None
    rect_mode = True
    current_label = cv2.GC_FGD
    start_point = (0, 0)
    mouse_pressed = False
    has_result = False
    image_tag = Path(image_name).stem

    print_grabcut_help()

    def render_preview() -> np.ndarray:
        preview = annotated.copy()
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
                cv2.circle(annotated, (x, y), 5, (255, 255, 255) if current_label == cv2.GC_FGD else (0, 0, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
            if rect_mode:
                rect = (min(start_point[0], x), min(start_point[1], y), abs(x - start_point[0]), abs(y - start_point[1]))
            else:
                cv2.circle(mask, (x, y), 5, current_label, -1)
                cv2.circle(annotated, (x, y), 5, (255, 255, 255) if current_label == cv2.GC_FGD else (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
            if rect_mode:
                rect = (min(start_point[0], x), min(start_point[1], y), abs(x - start_point[0]), abs(y - start_point[1]))
            else:
                cv2.circle(mask, (x, y), 5, current_label, -1)
                cv2.circle(annotated, (x, y), 5, (255, 255, 255) if current_label == cv2.GC_FGD else (0, 0, 255), -1)

    cv2.namedWindow(window_editor, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_editor, mouse_callback)

    while True:
        cv2.imshow(window_editor, render_preview())
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
            save_image(output_dir / f"08_grabcut_interactive_editor_{image_tag}.png", render_preview())
            save_image(output_dir / f"09_grabcut_interactive_result_{image_tag}.png", result)

    cv2.destroyAllWindows()


def main(
    show: bool = False,
    interactive: bool = True,
    kmeans_image_name: str | None = None,
    grabcut_image_name: str | None = None,
) -> None:
    assignment_dir = OUTPUT_DIR / "page68_assignment"
    assignment_dir.mkdir(parents=True, exist_ok=True)

    selected_kmeans = choose_image_name(
        kmeans_image_name,
        color_only=True,
        prompt="Select image for assignment K-means comparison:",
    )
    selected_grabcut = choose_image_name(
        grabcut_image_name,
        color_only=True,
        prompt="Select image for assignment GrabCut segmentation:",
    )

    kmeans_image = load_color_image(selected_kmeans)
    segmented_rgb = kmeans_segmentation(kmeans_image, include_xy=False)
    segmented_rgbxy = kmeans_segmentation(kmeans_image, include_xy=True)
    kmeans_tag = Path(selected_kmeans).stem

    save_image(assignment_dir / f"01_kmeans_original_{kmeans_tag}.png", kmeans_image)
    save_image(assignment_dir / f"02_kmeans_rgb_{kmeans_tag}.png", segmented_rgb)
    save_image(assignment_dir / f"03_kmeans_rgbxy_{kmeans_tag}.png", segmented_rgbxy)

    grabcut_image = load_color_image(selected_grabcut)
    initial_view, rect_result, refined_result = demo_grabcut(grabcut_image)
    grabcut_tag = Path(selected_grabcut).stem
    save_image(assignment_dir / f"04_grabcut_original_{grabcut_tag}.png", grabcut_image)
    save_image(assignment_dir / f"05_grabcut_rect_overlay_{grabcut_tag}.png", initial_view)
    save_image(assignment_dir / f"06_grabcut_after_rect_{grabcut_tag}.png", rect_result)
    save_image(assignment_dir / f"07_grabcut_refined_demo_{grabcut_tag}.png", refined_result)

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    views = [
        ("K-means original", kmeans_image),
        ("RGB only", segmented_rgb),
        ("RGB + XY", segmented_rgbxy),
        ("GrabCut original", grabcut_image),
        ("GrabCut after rect", rect_result),
        ("GrabCut refined", refined_result),
    ]

    for ax, (title, image) in zip(axes.flat, views):
        plot_bgr(ax, image, title)

    finalize_figure(fig, f"page68_assignment_summary_{kmeans_tag}_{grabcut_tag}.png", show)

    if interactive:
        run_interactive_grabcut(grabcut_image, assignment_dir, selected_grabcut)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 68 - Assignment solution")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--kmeans-image", help="Optional image name for the K-means comparison.")
    parser.add_argument("--grabcut-image", help="Optional image name for the GrabCut task.")
    parser.add_argument("--demo-only", action="store_true", help="Skip the interactive GrabCut editor.")
    args = parser.parse_args()
    main(
        show=args.show,
        interactive=not args.demo_only,
        kmeans_image_name=args.kmeans_image,
        grabcut_image_name=args.grabcut_image,
    )
