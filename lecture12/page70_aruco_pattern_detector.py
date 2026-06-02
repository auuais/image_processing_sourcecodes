from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import ARUCO_DICTIONARY_NAME, ARUCO_MARKER_IDS, aruco_scene_path, draw_text_block, finalize_figure, load_color_image, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 12 page 70: ArUco pattern detector.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = load_color_image(aruco_scene_path())
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY_NAME))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, rejected = detector.detectMarkers(gray)

    detected_view = scene.copy()
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(detected_view, corners, ids)

    rejected_view = scene.copy()
    for candidate in rejected:
        polyline = np.int32(candidate.reshape(-1, 1, 2))
        cv2.polylines(rejected_view, [polyline], True, (0, 0, 255), 2, cv2.LINE_AA)

    detected_ids = sorted(int(marker_id) for marker_id in ids.reshape(-1)) if ids is not None else []
    centers = []
    if ids is not None:
        for marker_id, marker_corners in zip(ids.reshape(-1), corners):
            center = marker_corners[0].mean(axis=0)
            centers.append(f"id {int(marker_id)} -> ({center[0]:.1f}, {center[1]:.1f})")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_bgr(axes[0, 0], scene, "Synthetic ArUco scene")
    plot_bgr(axes[0, 1], detected_view, f"Detected markers ({len(detected_ids)})")
    plot_bgr(axes[1, 0], rejected_view, f"Rejected candidates ({len(rejected)})")

    summary = (
        f"Dictionary: {ARUCO_DICTIONARY_NAME}\n"
        f"Expected IDs: {ARUCO_MARKER_IDS}\n"
        f"Detected IDs: {detected_ids}\n"
        f"Rejected candidates: {len(rejected)}\n\n"
        "Marker centers:\n"
        f"{chr(10).join(centers or ['none'])}\n\n"
        "The scene is generated locally so the detector output is\n"
        "fully reproducible from a clean checkout."
    )
    draw_text_block(axes[1, 1], "ArUco Summary", summary)
    finalize_figure(fig, "page70_aruco_pattern_detector.png", show=args.show)


if __name__ == "__main__":
    main()
