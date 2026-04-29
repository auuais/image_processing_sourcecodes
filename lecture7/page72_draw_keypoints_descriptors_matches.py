from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import detect_fast_keypoints, finalize_figure, load_color_image, parse_show_flag, plot_bgr, top_keypoints


def main(show: bool = False) -> None:
    image = load_color_image("scene01.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints = top_keypoints(detect_fast_keypoints(gray, threshold=160, nonmax_suppression=True), 120)
    rng = np.random.default_rng(0)
    for keypoint in keypoints:
        keypoint.size = 10.0 + 90.0 * float(rng.random())
        keypoint.angle = 360.0 * float(rng.random())

    match_subset = keypoints[:40]
    matches = [cv2.DMatch(i, i, 1.0) for i in range(len(match_subset))]

    simple_view = cv2.drawKeypoints(image, match_subset, None, (255, 0, 255))
    rich_view = cv2.drawKeypoints(
        image,
        match_subset,
        None,
        (0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    matches_view = cv2.drawMatches(
        image,
        match_subset,
        image,
        match_subset,
        matches,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    print(f"FAST keypoints used for draw demo: {len(match_subset)}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    plot_bgr(axes[0], simple_view, "Draw keypoints")
    plot_bgr(axes[1], rich_view, "Draw rich keypoints")
    plot_bgr(axes[2], matches_view, "Draw matches")
    finalize_figure(fig, "page72_draw_keypoints_descriptors_matches.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 72 - Draw keypoints, descriptors, and matches"))
