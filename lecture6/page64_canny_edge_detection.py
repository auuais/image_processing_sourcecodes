from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from common import choose_image_name, finalize_figure, load_color_image, plot_bgr, plot_gray


def main(show: bool = False, image_name: str | None = None) -> None:
    selected_name = choose_image_name(image_name, prompt="Select image for page 64 Canny edge detection:")
    image = load_color_image(selected_name)
    edges = cv2.Canny(image, 100, 200)
    image_tag = Path(selected_name).stem

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_bgr(axes[0], image, f"Original - {selected_name}")
    plot_gray(axes[1], edges, "Edges")

    finalize_figure(fig, f"page64_canny_edge_detection_{image_tag}.png", show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 64 - Canny edge detection")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--image", help="Optional image name from the data folder.")
    args = parser.parse_args()
    main(show=args.show, image_name=args.image)
