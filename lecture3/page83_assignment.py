from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUTPUT_DIR,
    finalize_figure,
    load_lena_color,
    normalize_channel,
    parse_show_flag,
    plot_bgr,
    plot_gray,
    save_image,
)


def gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    corrected = np.power(gray.astype(np.float32) / 255.0, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def save_assignment_images(output_dir: Path, images: dict[str, np.ndarray]) -> None:
    for name, image in images.items():
        save_image(output_dir / f"{name}.png", image)


def main(show: bool = False) -> None:
    color = load_lena_color()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    gray_gamma = gamma_correction(gray, gamma=0.5)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    h_display = normalize_channel(h_channel)
    s_display = normalize_channel(s_channel)
    v_display = normalize_channel(v_channel)

    h_median = cv2.medianBlur(h_channel, 7)
    s_gaussian = cv2.GaussianBlur(s_channel, (7, 7), 0)
    v_bilateral = cv2.bilateralFilter(v_channel, 9, 50, 50)

    h_median_display = normalize_channel(h_median)
    s_gaussian_display = normalize_channel(s_gaussian)
    v_bilateral_display = normalize_channel(v_bilateral)

    assignment_output_dir = OUTPUT_DIR / "page83_assignment"
    save_assignment_images(
        assignment_output_dir,
        {
            "01_color": color,
            "02_gray": gray,
            "03_gray_histogram_equalized": gray_equalized,
            "04_gray_gamma_corrected": gray_gamma,
            "05_h_channel_normalized": h_display,
            "06_s_channel_normalized": s_display,
            "07_v_channel_normalized": v_display,
            "08_h_median_filtered": h_median_display,
            "09_s_gaussian_filtered": s_gaussian_display,
            "10_v_bilateral_filtered": v_bilateral_display,
        },
    )

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    entries = [
        ("Original color", color, "bgr"),
        ("Grayscale", gray, "gray"),
        ("Gray + histogram equalization", gray_equalized, "gray"),
        ("Gray + gamma correction", gray_gamma, "gray"),
        ("H channel normalized", h_display, "gray"),
        ("S channel normalized", s_display, "gray"),
        ("V channel normalized", v_display, "gray"),
        ("H + median filter", h_median_display, "gray"),
        ("S + Gaussian filter", s_gaussian_display, "gray"),
        ("V + bilateral filter", v_bilateral_display, "gray"),
    ]

    for ax, (title, image, kind) in zip(axes.flat, entries):
        if kind == "bgr":
            plot_bgr(ax, image, title)
        else:
            plot_gray(ax, image, title)

    for ax in axes.flat[len(entries):]:
        ax.axis("off")

    finalize_figure(fig, "page83_assignment_summary.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 83 - Assignment solution"))
