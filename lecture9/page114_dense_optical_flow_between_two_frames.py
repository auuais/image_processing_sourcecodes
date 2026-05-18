from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, TRAFFIC_PATH, ensure_demo_data, finalize_figure, plot_bgr, plot_gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 9 page 114: dense optical flow on a traffic clip.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum number of video frames to process.")
    parser.add_argument("--show", action="store_true", help="Display processed frames while saving the outputs.")
    return parser.parse_args()


def draw_flow_overlay(image: np.ndarray, flow: np.ndarray, stride: int = 24, scale: float = 6.0) -> np.ndarray:
    overlay = image.copy()
    height, width = flow.shape[:2]
    for y in range(stride // 2, height, stride):
        for x in range(stride // 2, width, stride):
            dx, dy = flow[y, x]
            magnitude = float(np.hypot(dx, dy))
            if not 1.5 <= magnitude <= 15.0:
                continue
            end = (int(round(x + scale * dx)), int(round(y + scale * dy)))
            cv2.arrowedLine(overlay, (x, y), end, (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.25)
    return overlay


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(flow, axis=2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def create_secondary_flow():
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
        return cv2.optflow.createOptFlow_DualTVL1(), "DualTVL1"
    if hasattr(cv2, "createOptFlow_DualTVL1"):
        return cv2.createOptFlow_DualTVL1(), "DualTVL1"
    if hasattr(cv2, "DISOpticalFlow_create"):
        return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM), "DIS fallback for DualTVL1"
    return None, "Farneback fallback only"


def main() -> None:
    args = parse_args()
    ensure_demo_data()
    secondary_flow, secondary_label = create_secondary_flow()

    capture = cv2.VideoCapture(str(TRAFFIC_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {TRAFFIC_PATH}")

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError("Could not read the first traffic frame.")

    first_frame = cv2.resize(first_frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape[:2]
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 15.0

    farneback_writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "page114_dense_optical_flow_farneback.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(fps, 15.0),
        (width * 2, height),
    )
    secondary_writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "page114_dense_optical_flow_secondary.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(fps, 15.0),
        (width * 2, height),
    )
    if not farneback_writer.isOpened() or not secondary_writer.isOpened():
        capture.release()
        farneback_writer.release()
        secondary_writer.release()
        raise OSError("Could not create dense optical flow output videos.")

    farneback_flow = None
    secondary_prev_gray = prev_gray.copy()
    secondary_prev_flow = None
    summary = {}
    frame_index = 0
    try:
        while frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if farneback_flow is None:
                farneback_flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    0.5,
                    5,
                    13,
                    10,
                    5,
                    1.1,
                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                )
            else:
                farneback_flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    farneback_flow,
                    0.5,
                    5,
                    13,
                    10,
                    5,
                    1.1,
                    cv2.OPTFLOW_USE_INITIAL_FLOW,
                )

            farneback_overlay = draw_flow_overlay(frame, farneback_flow)
            farneback_mag = flow_magnitude(farneback_flow)
            farneback_heat = cv2.applyColorMap(farneback_mag, cv2.COLORMAP_TURBO)
            farneback_writer.write(np.hstack((farneback_overlay, farneback_heat)))

            if secondary_flow is None:
                secondary_overlay = farneback_overlay.copy()
                secondary_mag = farneback_mag.copy()
            else:
                if secondary_label.startswith("DualTVL1") and hasattr(secondary_flow, "setUseInitialFlow"):
                    secondary_flow.setUseInitialFlow(secondary_prev_flow is not None)
                secondary_prev_flow = secondary_flow.calc(
                    secondary_prev_gray,
                    gray,
                    secondary_prev_flow if secondary_label.startswith("DualTVL1") else None,
                )
                secondary_overlay = draw_flow_overlay(frame, secondary_prev_flow)
                secondary_mag = flow_magnitude(secondary_prev_flow)
            secondary_heat = cv2.applyColorMap(secondary_mag, cv2.COLORMAP_TURBO)
            secondary_writer.write(np.hstack((secondary_overlay, secondary_heat)))

            if frame_index == min(40, args.max_frames - 1):
                summary = {
                    "farneback_overlay": farneback_overlay.copy(),
                    "farneback_mag": farneback_mag.copy(),
                    "secondary_overlay": secondary_overlay.copy(),
                    "secondary_mag": secondary_mag.copy(),
                }

            if args.show:
                cv2.imshow("page114_farneback", np.hstack((farneback_overlay, farneback_heat)))
                cv2.imshow("page114_secondary", np.hstack((secondary_overlay, secondary_heat)))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            prev_gray = gray
            secondary_prev_gray = gray.copy()
            frame_index += 1
    finally:
        capture.release()
        farneback_writer.release()
        secondary_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if not summary:
        raise RuntimeError("Dense optical flow processing did not produce a summary frame.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_bgr(axes[0, 0], summary["farneback_overlay"], "Farneback Flow Overlay")
    plot_gray(axes[0, 1], summary["farneback_mag"], "Farneback Magnitude")
    plot_bgr(axes[1, 0], summary["secondary_overlay"], f"{secondary_label} Overlay")
    plot_gray(axes[1, 1], summary["secondary_mag"], f"{secondary_label} Magnitude")
    finalize_figure(fig, "page114_dense_optical_flow_between_two_frames.png", show=args.show)
    print(f"Saved video: {OUTPUT_DIR / 'page114_dense_optical_flow_farneback.mp4'}")
    print(f"Saved video: {OUTPUT_DIR / 'page114_dense_optical_flow_secondary.mp4'}")


if __name__ == "__main__":
    main()
