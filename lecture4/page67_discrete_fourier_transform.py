from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, normalize_channel, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("camera.png").astype(np.float32) / 255.0

    fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(fft, axes=[0, 1])
    magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
    magnitude = np.log1p(magnitude)

    restored = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_gray(axes[0], image, "Original")
    plot_gray(axes[1], normalize_channel(magnitude), "DFT magnitude")
    plot_gray(axes[2], restored, "Restored")

    finalize_figure(fig, "page67_discrete_fourier_transform.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 67 - Discrete Fourier Transform"))
