from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, parse_show_flag, plot_bgr


def main(show: bool = False) -> None:
    shape = (480, 640, 3)

    white = np.full(shape, 255, dtype=np.uint8)
    red = np.full(shape, (0, 0, 255), dtype=np.uint8)
    black = np.zeros(shape, dtype=np.uint8)

    black_with_white_pixels = black.copy()
    for x in (160, 320, 480):
        black_with_white_pixels[240, x] = (255, 255, 255)

    blue_with_white_pixels = black_with_white_pixels.copy()
    blue_with_white_pixels[:, :, 0] = 255

    blue_with_white_line = blue_with_white_pixels.copy()
    blue_with_white_line[:, 320, :] = 255

    blue_with_red_block = blue_with_white_line.copy()
    blue_with_red_block[100:480, 100:200, 2] = 255

    examples = [
        ("White image", white),
        ("Red image", red),
        ("Black image", black),
        ("Black with white pixels", black_with_white_pixels),
        ("Blue with white pixels", blue_with_white_pixels),
        ("Blue with white line", blue_with_white_line),
        ("Blue with red block", blue_with_red_block),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (title, image) in zip(axes.flat, examples):
        plot_bgr(ax, image, title)

    for ax in axes.flat[len(examples):]:
        ax.axis("off")

    finalize_figure(fig, "page76_matrix_manipulating.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 76 - Matrix manipulating"))
