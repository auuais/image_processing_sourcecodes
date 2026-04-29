from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from common import (
    compute_descriptor,
    cross_check_matches,
    detect_surf_keypoints,
    draw_matches,
    finalize_figure,
    load_color_image,
    load_gray_image,
    load_video_frame_pair,
    plot_bgr,
    resolved_descriptor_label,
    save_image,
    traffic_video_exists,
    TRAFFIC_VIDEO_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 8 page 87 - finding correspondences between descriptors")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib window after saving the output.")
    parser.add_argument(
        "--use-traffic",
        action="store_true",
        help="Use data/traffic.mp4 instead of the default static image pair.",
    )
    parser.add_argument("--frame-a", type=int, default=None, help="First video frame index for traffic.mp4.")
    parser.add_argument("--frame-b", type=int, default=None, help="Second video frame index for traffic.mp4.")
    return parser.parse_args()


def load_inputs(use_traffic: bool, frame_a: int | None, frame_b: int | None):
    if use_traffic:
        image1, image2, meta = load_video_frame_pair(TRAFFIC_VIDEO_PATH, frame_a=frame_a, frame_b=frame_b)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        return image1, image2, gray1, gray2, meta

    image1 = load_color_image("match_scene_reference.png")
    image2 = load_color_image("match_scene_query.png")
    gray1 = load_gray_image("match_scene_reference.png")
    gray2 = load_gray_image("match_scene_query.png")
    return image1, image2, gray1, gray2, None


def main(show: bool = False, use_traffic: bool = False, frame_a: int | None = None, frame_b: int | None = None) -> None:
    image1, image2, gray1, gray2, meta = load_inputs(use_traffic, frame_a, frame_b)

    surf_points1 = detect_surf_keypoints(gray1, max_points=500)
    surf_points2 = detect_surf_keypoints(gray2, max_points=500)

    descriptor_panels = []
    for descriptor_name in ["SURF", "BRIEF", "ORB"]:
        points1, desc1 = compute_descriptor(descriptor_name, gray1, surf_points1)
        points2, desc2 = compute_descriptor(descriptor_name, gray2, surf_points2)
        matches = cross_check_matches(desc1, desc2, descriptor_name)
        descriptor_panels.append(
            (
                f"{resolved_descriptor_label(descriptor_name)} raw matches ({len(matches)})",
                draw_matches(image1, points1, image2, points2, matches, max_matches=35),
            )
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if meta is None:
        plot_bgr(axes[0, 0], image1, "Reference image")
        plot_bgr(axes[0, 1], image2, "Query image")
    else:
        plot_bgr(axes[0, 0], image1, f"traffic.mp4 frame {meta['frame_a']}")
        plot_bgr(axes[0, 1], image2, f"traffic.mp4 frame {meta['frame_b']}")
    plot_bgr(axes[1, 0], descriptor_panels[0][1], descriptor_panels[0][0])
    plot_bgr(axes[1, 1], descriptor_panels[1][1], descriptor_panels[1][0])

    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 7))
    plot_bgr(ax2, descriptor_panels[2][1], descriptor_panels[2][0])
    if meta is not None:
        save_image(Path("C:/Users/USER/Documents/course_Translations/Computer_vision/lecture8/output/page87_traffic_frame_a.png"), image1)
        save_image(Path("C:/Users/USER/Documents/course_Translations/Computer_vision/lecture8/output/page87_traffic_frame_b.png"), image2)
        finalize_figure(fig, "page87_finding_correspondences_traffic_overview.png", show=False)
        finalize_figure(fig2, "page87_finding_correspondences_traffic_orb.png", show=show)
    else:
        finalize_figure(fig, "page87_finding_correspondences_overview.png", show=False)
        finalize_figure(fig2, "page87_finding_correspondences_orb.png", show=show)


if __name__ == "__main__":
    args = parse_args()
    auto_use_traffic = args.use_traffic or traffic_video_exists()
    main(show=args.show, use_traffic=auto_use_traffic, frame_a=args.frame_a, frame_b=args.frame_b)
