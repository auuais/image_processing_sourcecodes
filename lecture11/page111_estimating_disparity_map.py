from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import disparity_pair_images, finalize_figure, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 11 page 111: estimating disparity map for stereo images.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def normalize_disparity(disparity: np.ndarray) -> np.ndarray:
    disparity = disparity.astype(np.float32)
    valid = disparity > disparity.min()
    if not np.any(valid):
        return np.zeros_like(disparity, dtype=np.uint8)
    values = disparity[valid]
    scaled = (disparity - values.min()) / max(values.max() - values.min(), 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    left_image, right_image = disparity_pair_images()
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    stereo_bm = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    disparity_bm = stereo_bm.compute(left_gray, right_gray).astype(np.float32) / 16.0

    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity_sgbm = stereo_sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0

    disparity_bm_display = normalize_disparity(disparity_bm)
    disparity_sgbm_display = normalize_disparity(disparity_sgbm)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_bgr(axes[0, 0], left_image, "Left Stereo Image")
    plot_bgr(axes[0, 1], right_image, "Right Stereo Image")
    axes[1, 0].imshow(disparity_bm_display, cmap="plasma")
    axes[1, 0].set_title("StereoBM Disparity")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(disparity_sgbm_display, cmap="plasma")
    axes[1, 1].set_title("StereoSGBM Disparity")
    axes[1, 1].axis("off")
    finalize_figure(fig, "page111_estimating_disparity_map.png", show=args.show)


if __name__ == "__main__":
    main()

