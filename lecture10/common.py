from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
PINHOLE_DIR = DATA_DIR / "pinhole_calib"
FISHEYE_DIR = DATA_DIR / "fisheyes"
STEREO_DIR = DATA_DIR / "stereo" / "case1"

PINHOLE_PATTERN = (9, 6)
FISHEYE_PATTERN = (8, 6)
STEREO_PATTERN = (9, 6)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

PINHOLE_SOURCE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
FISHEYE_SOURCE_IDS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
STEREO_SOURCE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

PINHOLE_RESULT_PATH = PINHOLE_DIR / "camera_mat.npy"
PINHOLE_DIST_PATH = PINHOLE_DIR / "dist_coefs.npy"
FISHEYE_RESULT_PATH = FISHEYE_DIR / "camera_mat.npy"
FISHEYE_DIST_PATH = FISHEYE_DIR / "dist_coefs.npy"
STEREO_RESULT_PATH = STEREO_DIR / "stereo_case1.npz"


@dataclass
class ChessboardSample:
    path: Path
    image: np.ndarray
    gray: np.ndarray
    corners: np.ndarray
    visualized: np.ndarray


@dataclass
class MonoCalibration:
    pattern_size: tuple[int, int]
    samples: list[ChessboardSample]
    object_points: list[np.ndarray]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    rms: float


@dataclass
class StereoCalibration:
    pattern_size: tuple[int, int]
    left_samples: list[ChessboardSample]
    right_samples: list[ChessboardSample]
    object_points: list[np.ndarray]
    left_camera_matrix: np.ndarray
    left_dist_coeffs: np.ndarray
    right_camera_matrix: np.ndarray
    right_dist_coeffs: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    essential: np.ndarray
    fundamental: np.ndarray
    rms: float


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_float = image.astype(np.float32)
    if float(image_float.min()) >= 0.0 and float(image_float.max()) <= 1.0:
        image_float *= 255.0
    return np.clip(image_float, 0, 255).astype(np.uint8)


def ensure_color(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8(image)
    if image_uint8.ndim == 2:
        return cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    return image_uint8


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), to_uint8(image)):
        raise OSError(f"Could not save image: {path}")
    print(f"Saved image: {path}")
    return path


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(ensure_color(image), cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")


def plot_gray(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(to_uint8(image), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


def draw_text_block(ax, title: str, text: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)


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


def format_matrix(array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(array, precision=precision, suppress_small=True)


def download_image(url: str) -> np.ndarray:
    with urlopen(url, timeout=30) as response:
        payload = response.read()
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image from URL: {url}")
    return image


def write_downloaded_image(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(destination, download_image(url))
    return destination


def ensure_pinhole_dataset() -> None:
    for output_index, sample_id in enumerate(PINHOLE_SOURCE_IDS):
        write_downloaded_image(
            f"https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/left{sample_id:02d}.jpg",
            PINHOLE_DIR / f"img_{output_index:02d}.png",
        )


def ensure_fisheye_dataset() -> None:
    for output_index, sample_id in enumerate(FISHEYE_SOURCE_IDS):
        write_downloaded_image(
            "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/"
            f"cameracalibration/fisheye/calib-3_stereo_from_JY/left/stereo_pair_{sample_id:03d}.jpg",
            FISHEYE_DIR / f"Fisheye1_{output_index:02d}.png",
        )


def ensure_stereo_dataset() -> None:
    for sample_id in STEREO_SOURCE_IDS:
        write_downloaded_image(
            f"https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/left{sample_id:02d}.jpg",
            STEREO_DIR / f"left{sample_id:02d}.png",
        )
        write_downloaded_image(
            f"https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/right{sample_id:02d}.jpg",
            STEREO_DIR / f"right{sample_id:02d}.png",
        )


def ensure_demo_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_pinhole_dataset()
    ensure_fisheye_dataset()
    ensure_stereo_dataset()


def pinhole_image_paths() -> list[Path]:
    ensure_demo_data()
    return sorted(PINHOLE_DIR.glob("img_*.png"))


def fisheye_image_paths() -> list[Path]:
    ensure_demo_data()
    return sorted(FISHEYE_DIR.glob("Fisheye1_*.png"))


def stereo_image_paths() -> tuple[list[Path], list[Path]]:
    ensure_demo_data()
    return sorted(STEREO_DIR.glob("left*.png")), sorted(STEREO_DIR.glob("right*.png"))


def find_chessboard_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        return True, corners
    if hasattr(cv2, "findChessboardCornersSB"):
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern_size)
        if found_sb:
            return True, corners_sb.reshape(-1, 1, 2).astype(np.float32)
    return False, None


def collect_chessboard_samples(paths: list[Path], pattern_size: tuple[int, int]) -> list[ChessboardSample]:
    samples: list[ChessboardSample] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = find_chessboard_corners(gray, pattern_size)
        if not found or corners is None:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        visualized = image.copy()
        cv2.drawChessboardCorners(visualized, pattern_size, corners, True)
        samples.append(ChessboardSample(path, image, gray, corners, visualized))
    if not samples:
        raise RuntimeError(f"No valid chessboard detections found for pattern size {pattern_size}.")
    return samples


def build_object_points(pattern_size: tuple[int, int]) -> np.ndarray:
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    return object_points


def calibrate_pinhole() -> MonoCalibration:
    samples = collect_chessboard_samples(pinhole_image_paths(), PINHOLE_PATTERN)
    object_pattern = build_object_points(PINHOLE_PATTERN)
    object_points = [object_pattern.copy() for _ in samples]
    image_points = [sample.corners for sample in samples]
    image_size = samples[0].gray.shape[::-1]
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    np.save(PINHOLE_RESULT_PATH, camera_matrix)
    np.save(PINHOLE_DIST_PATH, dist_coeffs)
    print(f"Saved array: {PINHOLE_RESULT_PATH}")
    print(f"Saved array: {PINHOLE_DIST_PATH}")
    return MonoCalibration(PINHOLE_PATTERN, samples, object_points, camera_matrix, dist_coeffs, rvecs, tvecs, rms)


def calibrate_fisheye() -> MonoCalibration:
    samples = collect_chessboard_samples(fisheye_image_paths(), FISHEYE_PATTERN)
    object_pattern = build_object_points(FISHEYE_PATTERN).reshape(1, -1, 3)
    object_points = [object_pattern.copy() for _ in samples]
    image_points = [sample.corners.reshape(1, -1, 2) for sample in samples]
    image_size = samples[0].gray.shape[::-1]

    camera_matrix = np.zeros((3, 3))
    dist_coeffs = np.zeros((4, 1))
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        None,
        None,
        flags,
        SUBPIX_CRITERIA,
    )
    np.save(FISHEYE_RESULT_PATH, camera_matrix)
    np.save(FISHEYE_DIST_PATH, dist_coeffs)
    print(f"Saved array: {FISHEYE_RESULT_PATH}")
    print(f"Saved array: {FISHEYE_DIST_PATH}")
    return MonoCalibration(FISHEYE_PATTERN, samples, object_points, camera_matrix, dist_coeffs, rvecs, tvecs, rms)


def calibrate_stereo() -> StereoCalibration:
    left_paths, right_paths = stereo_image_paths()
    left_samples = collect_chessboard_samples(left_paths, STEREO_PATTERN)
    right_samples = collect_chessboard_samples(right_paths, STEREO_PATTERN)
    pair_count = min(len(left_samples), len(right_samples))
    left_samples = left_samples[:pair_count]
    right_samples = right_samples[:pair_count]

    object_pattern = build_object_points(STEREO_PATTERN)
    object_points = [object_pattern.copy() for _ in range(pair_count)]
    left_points = [sample.corners for sample in left_samples]
    right_points = [sample.corners for sample in right_samples]
    image_size = left_samples[0].gray.shape[::-1]

    rms, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        None,
        None,
        None,
        None,
        image_size,
    )
    np.savez(
        STEREO_RESULT_PATH,
        K1=left_camera_matrix,
        D1=left_dist_coeffs,
        K2=right_camera_matrix,
        D2=right_dist_coeffs,
        R=rotation,
        T=translation,
        E=essential,
        F=fundamental,
    )
    print(f"Saved array: {STEREO_RESULT_PATH}")
    return StereoCalibration(
        STEREO_PATTERN,
        left_samples,
        right_samples,
        object_points,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        rotation,
        translation,
        essential,
        fundamental,
        rms,
    )
