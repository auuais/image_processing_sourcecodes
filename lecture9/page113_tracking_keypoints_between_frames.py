from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, TRAFFIC_PATH, ensure_demo_data, finalize_figure, plot_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 9 page 113: Lucas-Kanade tracking between frames.")
    parser.add_argument("--show", action="store_true", help="Display the processed frames while saving the result.")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum number of video frames to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_demo_data()
    capture = cv2.VideoCapture(str(TRAFFIC_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {TRAFFIC_PATH}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / "page113_tracking_keypoints_between_frames.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_path}")

    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 5,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    prev_gray = None
    prev_pts = None
    track_mask = None
    first_frame = None
    summary_frame = None
    processed = 0
    try:
        while processed < args.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if first_frame is None:
                first_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if track_mask is None:
                track_mask = np.zeros_like(frame)

            if prev_pts is None or len(prev_pts) < 25:
                prev_pts = cv2.goodFeaturesToTrack(gray, 500, 0.05, 10)
                prev_gray = gray
                writer.write(frame)
                processed += 1
                continue

            pts, status, _errors = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            if pts is None or status is None:
                prev_pts = None
                prev_gray = gray
                writer.write(frame)
                processed += 1
                continue

            good_new = pts[status.reshape(-1) == 1].reshape(-1, 2)
            good_prev = prev_pts[status.reshape(-1) == 1].reshape(-1, 2)
            if len(good_new) == 0:
                prev_pts = None
                prev_gray = gray
                writer.write(frame)
                processed += 1
                continue

            for new_pt, old_pt in zip(good_new, good_prev):
                start = tuple(np.round(old_pt).astype(int))
                end = tuple(np.round(new_pt).astype(int))
                cv2.line(track_mask, start, end, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, end, 3, (0, 255, 0), -1, cv2.LINE_AA)

            annotated = cv2.addWeighted(frame, 1.0, track_mask, 0.65, 0.0)
            cv2.putText(annotated, f"Tracked points: {len(good_new)}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            writer.write(annotated)

            if processed == min(60, args.max_frames - 1):
                summary_frame = annotated.copy()

            if args.show:
                cv2.imshow("page113_tracking", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("c"):
                    track_mask.fill(0)
                    prev_pts = None

            prev_gray = gray
            prev_pts = good_new.reshape(-1, 1, 2)
            processed += 1
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if first_frame is None:
        raise RuntimeError("No frames were processed from the traffic video.")
    if summary_frame is None:
        summary_frame = first_frame.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_bgr(axes[0], first_frame, "Reference Frame")
    plot_bgr(axes[1], summary_frame, "Lucas-Kanade Tracking")
    finalize_figure(fig, "page113_tracking_keypoints_between_frames.png", show=args.show)
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
