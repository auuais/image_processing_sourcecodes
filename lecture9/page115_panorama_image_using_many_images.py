from __future__ import annotations

import argparse
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, create_stitcher, finalize_figure, panorama_image_paths, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 9 page 115: panorama image using many images.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = panorama_image_paths()
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read panorama image: {path}")
        images.append(image)

    stitcher = create_stitcher()
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK or panorama is None:
        raise RuntimeError(f"Panorama stitching failed with status code: {status}")

    max_panorama_width = 3200
    if panorama.shape[1] > max_panorama_width:
        scale = max_panorama_width / float(panorama.shape[1])
        panorama = cv2.resize(panorama, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    output_path = OUTPUT_DIR / "page115_panorama_result.jpg"
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not save panorama: {output_path}")
    print(f"Saved image: {output_path}")

    panel_count = len(images) + 1
    columns = 3
    rows = math.ceil(panel_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.8 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, path, image in zip(axes_flat, image_paths, images):
        plot_bgr(axis, image, path.stem)

    plot_bgr(axes_flat[len(images)], panorama, "Stitched Panorama")
    for axis in axes_flat[panel_count:]:
        axis.axis("off")

    finalize_figure(fig, "page115_panorama_image_using_many_images.png", show=args.show)


if __name__ == "__main__":
    main()
