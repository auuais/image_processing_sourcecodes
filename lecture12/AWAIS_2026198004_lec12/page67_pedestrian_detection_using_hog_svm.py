from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from common import draw_text_block, finalize_figure, load_color_image, pedestrian_image_paths, plot_bgr


PEOPLE_DETECTOR = cv2.HOGDescriptor()
PEOPLE_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


@dataclass
class DetectionResult:
    name: str
    original: np.ndarray
    annotated: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    scores: list[float]
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 12 page 67: pedestrian detection using HOG-SVM.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def nms_indices(boxes: list[list[int]], scores: list[float], nms_threshold: float = 0.35) -> list[int]:
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=nms_threshold)
    if len(indices) == 0:
        return list(range(len(boxes)))
    return np.array(indices).reshape(-1).astype(int).tolist()


def detect_pedestrians(name: str, image: np.ndarray) -> DetectionResult:
    start_time = time.perf_counter()
    raw_boxes, raw_weights = PEOPLE_DETECTOR.detectMultiScale(
        image,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    boxes = [list(map(int, box)) for box in np.asarray(raw_boxes).reshape(-1, 4)] if len(raw_boxes) else []
    scores = [float(score) for score in np.asarray(raw_weights).reshape(-1)] if len(raw_weights) else []
    keep = nms_indices(boxes, scores)
    filtered_boxes = [tuple(boxes[index]) for index in keep]
    filtered_scores = [scores[index] for index in keep]

    annotated = image.copy()
    for index, ((x, y, width, height), score) in enumerate(zip(filtered_boxes, filtered_scores), start=1):
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (40, 220, 40), 3, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"P{index} {score:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 120, 30),
            2,
            cv2.LINE_AA,
        )
    return DetectionResult(name, image, annotated, filtered_boxes, filtered_scores, elapsed_ms)


def main() -> None:
    args = parse_args()
    results = [detect_pedestrians(path.stem, load_color_image(path)) for path in pedestrian_image_paths()]

    fig = plt.figure(figsize=(17, max(6, 4.4 * len(results))))
    grid = GridSpec(len(results), 3, figure=fig, width_ratios=[1.0, 1.0, 0.95])

    for row_index, result in enumerate(results):
        plot_bgr(fig.add_subplot(grid[row_index, 0]), result.original, f"{result.name}: input")
        plot_bgr(
            fig.add_subplot(grid[row_index, 1]),
            result.annotated,
            f"{result.name}: detections ({len(result.boxes)})",
        )

    summary_lines = ["Detector: OpenCV HOGDescriptor + default people SVM", "Parameters: winStride=(8,8), padding=(8,8), scale=1.05", ""]
    for result in results:
        mean_score = float(np.mean(result.scores)) if result.scores else 0.0
        summary_lines.extend(
            [
                f"{result.name}",
                f"  detections: {len(result.boxes)}",
                f"  mean score: {mean_score:.3f}",
                f"  runtime: {result.elapsed_ms:.1f} ms",
                f"  boxes: {result.boxes}",
                "",
            ]
        )
    draw_text_block(fig.add_subplot(grid[:, 2]), "HOG-SVM Summary", "\n".join(summary_lines).rstrip())
    finalize_figure(fig, "page67_pedestrian_detection_using_hog_svm.png", show=args.show)


if __name__ == "__main__":
    main()
