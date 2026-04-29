from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_lena_gray, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    gray_float = load_lena_gray().astype(np.float32) / 255.0

    gamma = 0.5
    corrected = np.power(gray_float, gamma)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_gray(axes[0], (gray_float * 255).astype(np.uint8), "Original grayscale")
    plot_gray(axes[1], (corrected * 255).astype(np.uint8), f"Gamma corrected (gamma={gamma})")

    finalize_figure(fig, "page80_gamma_correction.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 80 - Gamma correction"))
