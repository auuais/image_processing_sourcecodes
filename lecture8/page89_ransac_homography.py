from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import (
    compute_descriptor,
    detect_surf_keypoints,
    draw_matches,
    finalize_figure,
    find_homography_from_matches,
    load_color_image,
    load_gray_image,
    parse_show_flag,
    plot_bgr,
    ratio_test_matches,
    surf_label,
    warp_corners,
)


def main(show: bool = False) -> None:
    image1 = load_color_image("match_scene_reference.png")
    image2 = load_color_image("match_scene_query.png")
    gray1 = load_gray_image("match_scene_reference.png")
    gray2 = load_gray_image("match_scene_query.png")

    points1 = detect_surf_keypoints(gray1, max_points=500)
    points2 = detect_surf_keypoints(gray2, max_points=500)
    points1, desc1 = compute_descriptor("SURF", gray1, points1)
    points2, desc2 = compute_descriptor("SURF", gray2, points2)
    matches = ratio_test_matches(desc1, desc2, "SURF", ratio=0.75)

    homography, inlier_mask = find_homography_from_matches(points1, points2, matches)
    inlier_mask_list = [int(flag) for flag in inlier_mask.ravel().tolist()] if inlier_mask is not None else None
    inlier_view = draw_matches(image1, points1, image2, points2, matches, max_matches=50, matches_mask=inlier_mask_list)

    projected = image2.copy()
    if homography is not None:
        polygon = warp_corners(image1, homography)
        cv2.polylines(projected, [polygon.astype(int)], True, (0, 255, 255), 3, cv2.LINE_AA)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_bgr(axes[0, 0], image1, "Reference image")
    plot_bgr(axes[0, 1], image2, "Query image")
    plot_bgr(axes[1, 0], inlier_view, f"{surf_label()} + ratio test + RANSAC")
    plot_bgr(axes[1, 1], projected, "Projected reference corners on query")
    finalize_figure(fig, "page89_ransac_homography.png", show=show)


if __name__ == "__main__":
    main(show=parse_show_flag("Lecture 8 page 89 - model based fitting using RANSAC"))
