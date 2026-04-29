from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUTPUT_DIR,
    finalize_figure,
    load_gray_image,
    normalize_channel,
    plot_gray,
    save_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Page 72 assignment solution")
    parser.add_argument("--show", action="store_true", help="Show the saved Matplotlib figures.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a trackbar window for the difference threshold step.",
    )
    return parser.parse_args()


def build_unsharp_kernel(ksize: int = 11, alpha: float = 2.0) -> np.ndarray:
    gaussian_1d = cv2.getGaussianKernel(ksize, 0)
    kernel = -alpha * (gaussian_1d @ gaussian_1d.T)
    kernel[ksize // 2, ksize // 2] += 1.0 + alpha
    return kernel


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    return cv2.magnitude(dx, dy)


def normalized_gabor(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, ktype=cv2.CV_32F)
    kernel /= np.sqrt((kernel * kernel).sum())
    return cv2.filter2D(image, cv2.CV_32F, kernel)


def circular_low_pass_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    height, width = shape
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
    return mask.astype(np.float32)


def square_low_pass_mask(shape: tuple[int, int], half_size: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)
    cy, cx = height // 2, width // 2
    mask[cy - half_size:cy + half_size, cx - half_size:cx + half_size] = 1.0
    return mask


def apply_frequency_mask(image: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(fft, axes=[0, 1])
    mask = np.dstack([mask_2d, mask_2d])
    filtered_shift = fft_shift * mask
    filtered_fft = np.fft.ifftshift(filtered_shift, axes=[0, 1])
    restored = cv2.idft(filtered_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return restored


def threshold_difference(diff_image: np.ndarray, threshold_value: int) -> np.ndarray:
    _, binary = cv2.threshold(diff_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def run_interactive_threshold(diff_image: np.ndarray) -> None:
    window_name = "Difference threshold"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(_value: int) -> None:
        threshold_value = cv2.getTrackbarPos("threshold", window_name)
        binary = threshold_difference(diff_image, threshold_value)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        preview = np.hstack([binary, opened, closed])
        cv2.imshow(window_name, preview)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold", window_name, 80, 255, update)
    update(80)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def save_assignment_images(output_dir: Path, images: dict[str, np.ndarray]) -> None:
    for name, image in images.items():
        save_image(output_dir / f"{name}.png", image)


def main(show: bool = False, interactive: bool = False) -> None:
    image = load_gray_image("camera.png")
    image_float = image.astype(np.float32) / 255.0

    unsharp = cv2.filter2D(image_float, -1, build_unsharp_kernel())
    sobel = sobel_magnitude(unsharp)
    gabor = normalized_gabor(unsharp)

    sobel_display = normalize_channel(sobel)
    gabor_display = normalize_channel(gabor)
    difference = cv2.absdiff(sobel_display, gabor_display)
    thresholded = threshold_difference(difference, 80)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, morph_kernel)
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, morph_kernel)

    fft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(fft, axes=[0, 1])
    dft_magnitude = normalize_channel(np.log1p(cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1])))

    circle_mask = circular_low_pass_mask(image.shape, radius=40)
    square_mask = square_low_pass_mask(image.shape, half_size=30)
    circle_filtered = apply_frequency_mask(image_float, circle_mask)
    square_filtered = apply_frequency_mask(image_float, square_mask)

    assignment_dir = OUTPUT_DIR / "page72_assignment"
    save_assignment_images(
        assignment_dir,
        {
            "01_original": image,
            "02_unsharp": unsharp,
            "03_sobel_on_unsharp": sobel_display,
            "04_gabor_on_unsharp": gabor_display,
            "05_difference": difference,
            "06_difference_thresholded": thresholded,
            "07_opening": opened,
            "08_closing": closed,
            "09_dft_magnitude": dft_magnitude,
            "10_circle_mask": circle_mask * 255,
            "11_circle_filtered": circle_filtered,
            "12_square_mask": square_mask * 255,
            "13_square_filtered": square_filtered,
        },
    )

    fig_image_filter, axes = plt.subplots(3, 3, figsize=(12, 12))
    image_entries = [
        ("Original", image),
        ("Unsharp mask", unsharp),
        ("Sobel on (1)", sobel_display),
        ("Gabor on (1)", gabor_display),
        ("|Sobel - Gabor|", difference),
        ("Thresholded difference", thresholded),
        ("Opening on (4)", opened),
        ("Closing on (4)", closed),
    ]

    for ax, (title, result) in zip(axes.flat, image_entries):
        plot_gray(ax, result, title)

    for ax in axes.flat[len(image_entries):]:
        ax.axis("off")

    finalize_figure(fig_image_filter, "page72_assignment_image_filtering.png", show)

    fig_frequency, axes = plt.subplots(2, 3, figsize=(12, 8))
    frequency_entries = [
        ("Original", image),
        ("DFT magnitude", dft_magnitude),
        ("Circle mask", circle_mask * 255),
        ("Circle filtered", circle_filtered),
        ("Square mask", square_mask * 255),
        ("Square filtered", square_filtered),
    ]

    for ax, (title, result) in zip(axes.flat, frequency_entries):
        plot_gray(ax, result, title)

    finalize_figure(fig_frequency, "page72_assignment_frequency_filtering.png", show)

    if interactive:
        run_interactive_threshold(difference)


if __name__ == "__main__":
    args = parse_args()
    main(show=args.show, interactive=args.interactive)
