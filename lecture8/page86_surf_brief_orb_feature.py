from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from common import (
    compute_descriptor,
    create_surf_detector,
    detect_and_compute_orb,
    detect_surf_keypoints,
    ensure_color,
    finalize_figure,
    load_color_image,
    load_gray_image,
    parse_show_flag,
    plot_bgr,
    resolved_descriptor_label,
    surf_label,
)


def draw_keypoints(image, keypoints, color):
    return cv2.drawKeypoints(
        ensure_color(image),
        keypoints,
        None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def main(show: bool = False) -> None:
    color = load_color_image("match_scene_reference.png")
    gray = load_gray_image("match_scene_reference.png")

    surf_keypoints = detect_surf_keypoints(gray, max_points=350)
    surf_keypoints, surf_descriptors = compute_descriptor("SURF", gray, surf_keypoints)
    brief_keypoints, brief_descriptors = compute_descriptor("BRIEF", gray, surf_keypoints)
    orb_keypoints, orb_descriptors = detect_and_compute_orb(gray)
    orb_keypoints = orb_keypoints[:350]
    orb_descriptors = orb_descriptors[: len(orb_keypoints)] if orb_descriptors is not None else None

    surf_view = draw_keypoints(color, surf_keypoints, (0, 255, 0))
    orb_view = draw_keypoints(color, orb_keypoints, (0, 0, 255))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_bgr(axes[0, 0], color, "Reference scene")
    plot_bgr(axes[0, 1], surf_view, surf_label())
    plot_bgr(axes[1, 0], orb_view, "ORB detector + descriptor")
    axes[1, 1].axis("off")

    stats = [
        f"SURF detector backend: {surf_label()}",
        f"SURF/SIFT keypoints: {len(surf_keypoints)}",
        f"SURF descriptor shape: {None if surf_descriptors is None else surf_descriptors.shape}",
        f"{resolved_descriptor_label('BRIEF')} keypoints: {len(brief_keypoints)}",
        f"{resolved_descriptor_label('BRIEF')} descriptor shape: {None if brief_descriptors is None else brief_descriptors.shape}",
        f"ORB keypoints: {len(orb_keypoints)}",
        f"ORB descriptor shape: {None if orb_descriptors is None else orb_descriptors.shape}",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(stats), va="top", ha="left", fontsize=11, family="monospace")
    axes[1, 1].set_title("Feature summary")

    finalize_figure(fig, "page86_surf_brief_orb_feature.png", show=show)


if __name__ == "__main__":
    main(show=parse_show_flag("Lecture 8 page 86 - SURF, BRIEF, ORB feature overview"))
