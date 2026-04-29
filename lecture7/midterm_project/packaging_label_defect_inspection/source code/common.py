from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUT_DIR = ROOT_DIR / "output"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
MATCHES_DIR = OUTPUT_DIR / "matches"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
MASKS_DIR = OUTPUT_DIR / "masks"
RESULTS_CSV = OUTPUT_DIR / "inspection_results.csv"
REPORT_MD = OUTPUT_DIR / "RESULTS.md"


def ensure_dirs() -> None:
    for path in [DATA_DIR, REFERENCE_DIR, SAMPLES_DIR, OUTPUT_DIR, SUMMARIES_DIR, MATCHES_DIR, ALIGNED_DIR, MASKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def ensure_color(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8(image)
    if image_uint8.ndim == 2:
        return cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    return image_uint8


def normalize_channel(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    min_value = float(image_float.min())
    max_value = float(image_float.max())

    if max_value - min_value < 1e-6:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image_float - min_value) * 255.0 / (max_value - min_value)
    return scaled.astype(np.uint8)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_float = image.astype(np.float32)
    if float(image_float.min()) >= 0.0 and float(image_float.max()) <= 1.0:
        return np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
    return normalize_channel(image_float)


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), to_uint8(image))
    if not success:
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


def finalize_figure(fig, filename: str) -> Path:
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved csv: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
