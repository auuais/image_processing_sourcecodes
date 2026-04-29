from __future__ import annotations

import matplotlib.pyplot as plt

from common import (
    compute_descriptor,
    cross_check_matches,
    detect_surf_keypoints,
    draw_matches,
    finalize_figure,
    intersection_matches,
    load_color_image,
    load_gray_image,
    parse_show_flag,
    plot_bgr,
    ratio_test_matches,
    surf_label,
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

    raw_matches = cross_check_matches(desc1, desc2, "SURF")
    ratio_matches = ratio_test_matches(desc1, desc2, "SURF", ratio=0.75)
    mutual_matches = cross_check_matches(desc1, desc2, "SURF")
    consensus_matches = intersection_matches(ratio_matches, mutual_matches)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_bgr(axes[0, 0], image1, "Reference image")
    plot_bgr(axes[0, 1], image2, "Query image")
    plot_bgr(axes[1, 0], draw_matches(image1, points1, image2, points2, raw_matches, max_matches=35), f"{surf_label()} cross-check ({len(raw_matches)})")
    plot_bgr(axes[1, 1], draw_matches(image1, points1, image2, points2, ratio_matches, max_matches=35), f"{surf_label()} ratio test ({len(ratio_matches)})")
    finalize_figure(fig, "page88_feature_matching_consistency_ratio.png", show=show)

    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 7))
    plot_bgr(ax2, draw_matches(image1, points1, image2, points2, consensus_matches, max_matches=45), f"{surf_label()} ratio + consistency ({len(consensus_matches)})")
    finalize_figure(fig2, "page88_feature_matching_consistency_crosscheck.png", show=False)


if __name__ == "__main__":
    main(show=parse_show_flag("Lecture 8 page 88 - feature matching with consistency check and ratio test"))
