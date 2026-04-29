from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    TRAFFIC_VIDEO_PATH,
    compute_descriptor,
    detect_and_compute_orb,
    detect_surf_keypoints,
    ratio_test_matches,
    resolved_descriptor_label,
    warp_corners,
)


EXCLUDED_REFERENCE_NAMES = {
    "match_scene_reference.png",
    "match_scene_query.png",
}
REFERENCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an annotated traffic video using external car reference crops.")
    parser.add_argument("--video", type=Path, default=TRAFFIC_VIDEO_PATH, help="Input traffic video path.")
    parser.add_argument("--descriptor", choices=["ORB", "SURF", "BRIEF"], default="SURF", help="Descriptor used for matching.")
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing cropped car reference images.",
    )
    parser.add_argument("--min-matches", type=int, default=8, help="Minimum ratio-test matches before homography.")
    parser.add_argument("--min-inliers", type=int, default=6, help="Minimum RANSAC inliers to mark a detection.")
    parser.add_argument("--output-name", default="page87_traffic_car_match_video.mp4", help="Output video file name.")
    return parser.parse_args()


def load_reference_images(references_dir: Path) -> list[tuple[Path, np.ndarray]]:
    search_dirs = [references_dir]
    nested_references_dir = references_dir / "references"
    if nested_references_dir.is_dir():
        search_dirs.append(nested_references_dir)

    loaded: list[tuple[Path, np.ndarray]] = []
    seen: set[Path] = set()
    for search_dir in search_dirs:
        for path in sorted(search_dir.iterdir()):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            if path.name in EXCLUDED_REFERENCE_NAMES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                loaded.append((path, image))

    if not loaded:
        raise FileNotFoundError(
            f"No reference crops found in {references_dir}. "
            "Place the cropped car images directly in that folder or in a 'references' subfolder."
        )
    return loaded


def detect_and_describe(gray: np.ndarray, descriptor_name: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if descriptor_name == "ORB":
        return detect_and_compute_orb(gray)
    keypoints = detect_surf_keypoints(gray, max_points=700)
    return compute_descriptor(descriptor_name, gray, keypoints)


def reference_bundle(reference_images: list[tuple[Path, np.ndarray]], descriptor_name: str) -> list[dict[str, object]]:
    bundle: list[dict[str, object]] = []
    for idx, (path, image) in enumerate(reference_images, start=1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detect_and_describe(gray, descriptor_name)
        if descriptors is None or len(keypoints) < 4:
            continue
        bundle.append(
            {
                "index": idx,
                "name": path.stem,
                "path": path,
                "frame": image,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )

    if not bundle:
        raise RuntimeError("Reference crops were found, but none produced enough descriptors for matching.")
    return bundle


def find_best_reference(frame: np.ndarray, descriptor_name: str, references: list[dict[str, object]], min_matches: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detect_and_describe(gray, descriptor_name)
    if descriptors is None or len(keypoints) < 4:
        return None

    best = None
    for ref in references:
        ref_descriptors = ref["descriptors"]
        ref_keypoints = ref["keypoints"]
        matches = ratio_test_matches(ref_descriptors, descriptors, descriptor_name, ratio=0.75)
        if len(matches) < min_matches:
            continue

        src = np.float32([ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        candidate = {
            "reference_index": ref["index"],
            "reference_name": ref["name"],
            "homography": homography,
            "matches": len(matches),
            "inliers": inliers,
            "reference_frame": ref["frame"],
        }
        if best is None or (candidate["inliers"], candidate["matches"]) > (best["inliers"], best["matches"]):
            best = candidate

    return best


def draw_reference_strip(frame: np.ndarray, references: list[dict[str, object]], best_reference_index: int | None) -> np.ndarray:
    output = frame.copy()
    thumb_w = 140
    thumb_h = 84
    gap = 10
    x = 10

    for ref in references:
        thumb = cv2.resize(ref["frame"], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        output[10 : 10 + thumb_h, x : x + thumb_w] = thumb
        color = (0, 255, 0) if ref["index"] == best_reference_index else (255, 255, 255)
        cv2.rectangle(output, (x, 10), (x + thumb_w, 10 + thumb_h), color, 2, cv2.LINE_AA)
        cv2.putText(output, str(ref["name"])[:18], (x + 6, 10 + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        x += thumb_w + gap

    return output


def annotate_frame(
    frame: np.ndarray,
    best,
    descriptor_name: str,
    min_inliers: int,
    frame_idx: int,
    references: list[dict[str, object]],
) -> np.ndarray:
    annotated = draw_reference_strip(frame, references, best["reference_index"] if best else None)
    status = "NO STRONG MATCH"
    status_color = (0, 0, 255)

    if best is not None and best["homography"] is not None and best["inliers"] >= min_inliers:
        polygon = warp_corners(best["reference_frame"], best["homography"])
        cv2.polylines(annotated, [polygon.astype(int)], True, (0, 255, 255), 3, cv2.LINE_AA)
        status = "MATCH DETECTED"
        status_color = (0, 255, 0)

    lines = [
        f"Descriptor: {resolved_descriptor_label(descriptor_name)}",
        f"Frame: {frame_idx}",
        f"Status: {status}",
    ]
    if best is not None:
        lines.append(f"Best ref: {best['reference_name']}  matches={best['matches']}  inliers={best['inliers']}")
    else:
        lines.append("Best ref: none  matches=0  inliers=0")

    y = annotated.shape[0] - 90
    for idx, text in enumerate(lines):
        cv2.putText(annotated, text, (20, y + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            annotated,
            text,
            (20, y + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            status_color if idx == 2 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Missing input video: {args.video}")

    reference_images = load_reference_images(args.references_dir)
    references = reference_bundle(reference_images, args.descriptor)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = OUTPUT_DIR / args.output_name
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 20.0, frame_size)
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {out_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            best = find_best_reference(frame, args.descriptor, references, args.min_matches)
            annotated = annotate_frame(frame, best, args.descriptor, args.min_inliers, frame_idx, references)
            writer.write(annotated)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    print(f"Saved video: {out_path}")
    print("Reference crops:")
    for ref in references:
        print(f"- {ref['name']} ({ref['path'].name})")
    print(f"Descriptor backend: {resolved_descriptor_label(args.descriptor)}")


if __name__ == "__main__":
    main()
