from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import detect_sift, finalize_figure, load_color_image, load_gray_image, parse_show_flag, plot_bgr, top_keypoints


def main(show: bool = False) -> None:
    image1 = load_color_image("scene01.png")
    image2 = load_color_image("scene02_moved.png")
    gray1 = load_gray_image("scene01.png")
    gray2 = load_gray_image("scene02_moved.png")

    keypoints1, _ = detect_sift(gray1, nfeatures=80)
    keypoints2, _ = detect_sift(gray2, nfeatures=80)
    keypoints1 = top_keypoints(keypoints1, 80)
    keypoints2 = top_keypoints(keypoints2, 80)

    view1 = cv2.drawKeypoints(
        image1,
        keypoints1,
        None,
        (0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    view2 = cv2.drawKeypoints(
        image2,
        keypoints2,
        None,
        (0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    print(f"SIFT keypoints on image 1: {len(keypoints1)}")
    print(f"SIFT keypoints on image 2: {len(keypoints2)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_bgr(axes[0], view1, "SIFT keypoints - scene 1")
    plot_bgr(axes[1], view2, "SIFT keypoints - scene 2")
    finalize_figure(fig, "page73_sift_scale_invariant_keypoints.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 73 - Detecting scale invariant keypoints"))
