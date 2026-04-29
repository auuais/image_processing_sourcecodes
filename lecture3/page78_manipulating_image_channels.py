from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_lena_color, parse_show_flag, plot_bgr


def main(show: bool = False) -> None:
    image = load_lena_color().astype(np.float32) / 255.0
    print(f"Shape: {image.shape}")

    swapped = image.copy()
    swapped[:, :, [0, 2]] = swapped[:, :, [2, 0]]

    adjusted = image.copy()
    adjusted[:, :, [0, 2]] = adjusted[:, :, [2, 0]]
    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] * 0.9, 0.0, 1.0)
    adjusted[:, :, 1] = np.clip(adjusted[:, :, 1] * 1.1, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_bgr(axes[0], image, "Original image")
    plot_bgr(axes[1], swapped, "Blue and red swapped")
    plot_bgr(axes[2], adjusted, "Adjusted channels")

    finalize_figure(fig, "page78_manipulating_image_channels.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 78 - Manipulating image channels"))
