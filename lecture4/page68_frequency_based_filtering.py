from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, normalize_channel, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("moon.png").astype(np.float32) / 255.0

    fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(fft, axes=[0, 1])

    sz = 25
    mask = np.zeros(fft_shift.shape, dtype=np.float32)
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    mask[cy - sz:cy + sz, cx - sz:cx + sz, :] = 1.0

    fft_shift_filtered = fft_shift * mask
    fft_filtered = np.fft.ifftshift(fft_shift_filtered, axes=[0, 1])
    filtered = cv2.idft(fft_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_gray(axes[0], image, "Original")
    plot_gray(axes[1], normalize_channel(np.log1p(cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1]))), "Shifted spectrum")
    plot_gray(axes[2], mask[:, :, 0] * 255, "Square low-pass mask")
    plot_gray(axes[3], filtered, "No high frequencies")

    finalize_figure(fig, "page68_frequency_based_filtering.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 68 - Frequency-based filtering"))
