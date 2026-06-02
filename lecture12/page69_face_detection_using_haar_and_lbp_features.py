from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from common import draw_text_block, face_scene_path, face_video_path, finalize_figure, haar_cascade_path, lbp_cascade_path, load_color_image, plot_bgr


@dataclass
class FaceDetectionResult:
    label: str
    annotated: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    elapsed_ms: float


@dataclass
class VideoFrameResult:
    frame_index: int
    timestamp_s: float
    original: np.ndarray
    haar: FaceDetectionResult
    lbp: FaceDetectionResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 12 page 69: face detection using Haar and LBP features.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def detect_faces(
    image,
    classifier: cv2.CascadeClassifier,
    label: str,
    color: tuple[int, int, int],
    scale_factor: float,
    min_neighbors: int,
    min_size: tuple[int, int],
    min_area: int = 0,
) -> FaceDetectionResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start_time = time.perf_counter()
    raw_boxes = classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    boxes = [tuple(map(int, box)) for box in raw_boxes]
    if min_area > 0:
        boxes = [box for box in boxes if box[2] * box[3] >= min_area]

    annotated = image.copy()
    for index, (x, y, width, height) in enumerate(boxes, start=1):
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"{label[0]}{index}",
            (x, max(28, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return FaceDetectionResult(label, annotated, boxes, elapsed_ms)


def total_box_area(boxes: list[tuple[int, int, int, int]]) -> int:
    return sum(width * height for _, _, width, height in boxes)


def select_video_frame_results(
    video_path,
    haar_classifier: cv2.CascadeClassifier,
    lbp_classifier: cv2.CascadeClassifier,
    max_frames: int = 3,
    step: int = 15,
    min_spacing: int = 45,
) -> list[VideoFrameResult]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates: list[VideoFrameResult] = []

    for frame_index in range(0, frame_count, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        haar = detect_faces(
            frame,
            haar_classifier,
            label="Haar",
            color=(50, 200, 50),
            scale_factor=1.10,
            min_neighbors=7,
            min_size=(60, 60),
            min_area=10000,
        )
        lbp = detect_faces(
            frame,
            lbp_classifier,
            label="LBP",
            color=(0, 120, 255),
            scale_factor=1.10,
            min_neighbors=7,
            min_size=(60, 60),
            min_area=10000,
        )
        total_faces = len(haar.boxes) + len(lbp.boxes)
        if total_faces == 0:
            continue
        candidates.append(
            VideoFrameResult(
                frame_index=frame_index,
                timestamp_s=frame_index / fps,
                original=frame,
                haar=haar,
                lbp=lbp,
            )
        )
    capture.release()

    ranked = sorted(
        candidates,
        key=lambda item: (
            len(item.haar.boxes) + len(item.lbp.boxes),
            total_box_area(item.haar.boxes) + total_box_area(item.lbp.boxes),
        ),
        reverse=True,
    )
    selected: list[VideoFrameResult] = []
    for candidate in ranked:
        if all(abs(candidate.frame_index - chosen.frame_index) >= min_spacing for chosen in selected):
            selected.append(candidate)
        if len(selected) == max_frames:
            break

    if len(selected) < max_frames:
        for candidate in sorted(candidates, key=lambda item: item.frame_index):
            if candidate.frame_index in {chosen.frame_index for chosen in selected}:
                continue
            selected.append(candidate)
            if len(selected) == max_frames:
                break
    return sorted(selected, key=lambda item: item.frame_index)


def main() -> None:
    args = parse_args()
    haar_classifier = cv2.CascadeClassifier(str(haar_cascade_path()))
    lbp_classifier = cv2.CascadeClassifier(str(lbp_cascade_path()))
    video_path = face_video_path()

    if video_path is not None:
        frame_results = select_video_frame_results(video_path, haar_classifier, lbp_classifier)
        fig = plt.figure(figsize=(18, 14))
        grid = GridSpec(4, len(frame_results), figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.42])

        for column, result in enumerate(frame_results):
            title_suffix = f"f={result.frame_index}, t={result.timestamp_s:.2f}s"
            plot_bgr(fig.add_subplot(grid[0, column]), result.original, f"Video frame\n{title_suffix}")
            plot_bgr(
                fig.add_subplot(grid[1, column]),
                result.haar.annotated,
                f"Haar ({len(result.haar.boxes)})\n{title_suffix}",
            )
            plot_bgr(
                fig.add_subplot(grid[2, column]),
                result.lbp.annotated,
                f"LBP ({len(result.lbp.boxes)})\n{title_suffix}",
            )

        summary_lines = [
            f"Input video: {video_path.name}",
            f"Haar cascade: {haar_cascade_path().name}",
            f"LBP cascade: {lbp_cascade_path().name}",
            "",
        ]
        for result in frame_results:
            summary_lines.extend(
                [
                    f"frame {result.frame_index} ({result.timestamp_s:.2f}s)",
                    f"  Haar: {len(result.haar.boxes)} faces, {result.haar.elapsed_ms:.1f} ms",
                    f"  LBP: {len(result.lbp.boxes)} faces, {result.lbp.elapsed_ms:.1f} ms",
                    "",
                ]
            )
        draw_text_block(fig.add_subplot(grid[3, :]), "Face Detection Summary", "\n".join(summary_lines).rstrip())
    else:
        image = load_color_image(face_scene_path())
        haar_result = detect_faces(
            image,
            haar_classifier,
            label="Haar",
            color=(50, 200, 50),
            scale_factor=1.10,
            min_neighbors=4,
            min_size=(60, 60),
        )
        lbp_result = detect_faces(
            image,
            lbp_classifier,
            label="LBP",
            color=(0, 120, 255),
            scale_factor=1.05,
            min_neighbors=5,
            min_size=(60, 60),
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plot_bgr(axes[0, 0], image, "Synthetic multi-face scene")
        plot_bgr(axes[0, 1], haar_result.annotated, f"Haar detections ({len(haar_result.boxes)})")
        plot_bgr(axes[1, 0], lbp_result.annotated, f"LBP detections ({len(lbp_result.boxes)})")

        summary = (
            "Input: generated scene built from the official lena.jpg sample\n"
            "Goal: compare classical frontal-face cascade detectors\n\n"
            f"Haar cascade runtime: {haar_result.elapsed_ms:.1f} ms\n"
            f"Haar boxes: {haar_result.boxes}\n\n"
            f"LBP cascade runtime: {lbp_result.elapsed_ms:.1f} ms\n"
            f"LBP boxes: {lbp_result.boxes}\n\n"
            "The scene uses three frontal faces at different scales so\n"
            "both cascades can be compared on the same target layout."
        )
        draw_text_block(axes[1, 1], "Face Detection Summary", summary)
    finalize_figure(fig, "page69_face_detection_using_haar_and_lbp_features.png", show=args.show)


if __name__ == "__main__":
    main()
