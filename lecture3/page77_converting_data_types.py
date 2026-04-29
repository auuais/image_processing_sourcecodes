from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_lena_color, parse_show_flag, plot_bgr


def main(show: bool = False) -> None:
    image = load_lena_color()
    print(f"Original shape: {image.shape}, dtype: {image.dtype}")

    float_image = image.astype(np.float32) / 255.0
    print(f"Float shape: {float_image.shape}, dtype: {float_image.dtype}")

    brightened = np.clip(float_image * 2.0, 0.0, 1.0)
    restored = np.clip(float_image * 255.0, 0, 255).astype(np.uint8)
    print(f"Restored shape: {restored.shape}, dtype: {restored.dtype}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_bgr(axes[0], image, "Original uint8 image")
    plot_bgr(axes[1], brightened, "Float32 image x 2")
    plot_bgr(axes[2], restored, "Converted back to uint8")

    finalize_figure(fig, "page77_converting_data_types.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 77 - Converting data types"))
