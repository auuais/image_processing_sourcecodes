from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"


def parse_show_flag(description: str) -> bool:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib window after saving the output.",
    )
    return parser.parse_args().show


def load_color_image(name: str) -> np.ndarray:
    image_path = DATA_DIR / name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def load_gray_image(name: str) -> np.ndarray:
    image_path = DATA_DIR / name
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def load_affine_matrix(name: str = "scene02_affine.txt") -> np.ndarray:
    matrix_path = DATA_DIR / name
    if not matrix_path.exists():
        raise FileNotFoundError(f"Could not read affine matrix: {matrix_path}")
    matrix = np.loadtxt(matrix_path, dtype=np.float32)
    return matrix.reshape(2, 3)


def normalize_channel(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    min_value = float(image_float.min())
    max_value = float(image_float.max())

    if max_value - min_value < 1e-6:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (image_float - min_value) * 255.0 / (max_value - min_value)
    return normalized.astype(np.uint8)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    if float(image_float.min()) >= 0.0 and float(image_float.max()) <= 1.0:
        return np.clip(image_float * 255.0, 0, 255).astype(np.uint8)

    return normalize_channel(image_float)


def ensure_color(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8(image)
    if image_uint8.ndim == 2:
        return cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    return image_uint8


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(ensure_color(image), cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")


def plot_gray(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(to_uint8(image), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


def finalize_figure(fig, filename: str, show: bool = False) -> Path:
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), to_uint8(image))
    if not success:
        raise OSError(f"Could not save image: {path}")
    print(f"Saved image: {path}")
    return path


def top_keypoints(keypoints: list[cv2.KeyPoint], max_points: int) -> list[cv2.KeyPoint]:
    if len(keypoints) <= max_points:
        return list(keypoints)
    ranked = sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True)
    return ranked[:max_points]


def keypoints_to_points(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)


def points_to_keypoints(points: np.ndarray, size: float = 5.0) -> list[cv2.KeyPoint]:
    if points.size == 0:
        return []
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in points]


def detect_fast_keypoints(
    gray: np.ndarray,
    threshold: int = 30,
    nonmax_suppression: bool = True,
    detector_type: int = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
) -> list[cv2.KeyPoint]:
    detector = cv2.FastFeatureDetector_create(threshold, nonmax_suppression, detector_type)
    return list(detector.detect(gray))


def detect_gftt_points(
    gray: np.ndarray,
    max_corners: int = 100,
    quality_level: float = 0.05,
    min_distance: float = 10.0,
) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def detect_harris_points(
    gray: np.ndarray,
    max_corners: int = 150,
    quality_level: float = 0.01,
    min_distance: float = 5.0,
    k: float = 0.04,
) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        gray,
        max_corners,
        quality_level,
        min_distance,
        useHarrisDetector=True,
        k=k,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def create_sift(nfeatures: int = 0):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=nfeatures)
    raise RuntimeError("This OpenCV build does not provide SIFT_create().")


def detect_sift(gray: np.ndarray, nfeatures: int = 80) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    detector = create_sift(nfeatures=nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return list(keypoints), descriptors
