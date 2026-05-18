from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import LENA_PATH, finalize_figure, plot_bgr, read_color, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 9 page 112: remapping using an arbitrary transformation.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_color(LENA_PATH)
    height, width = image.shape[:2]

    x_coords = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    y_coords = np.tile(np.arange(height, dtype=np.float32).reshape(-1, 1), (1, width))

    xmap = x_coords + 30.0 * np.cos(20.0 * x_coords / max(height, 1))
    ymap = y_coords + 30.0 * np.sin(20.0 * y_coords / max(width, 1))

    remapped = cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    save_image(LENA_PATH.parent.parent / "output" / "page112_remapped_image.png", remapped)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_bgr(axes[0], image, "Original Image")
    plot_bgr(axes[1], remapped, "Remapped Image")
    finalize_figure(fig, "page112_remapping_arbitrary_transformation.png", show=args.show)


if __name__ == "__main__":
    main()
