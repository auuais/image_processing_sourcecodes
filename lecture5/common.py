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
        help="Open the generated Matplotlib window after saving the output.",
    )
    return parser.parse_args().show


def load_gray_image(name: str) -> np.ndarray:
    image_path = DATA_DIR / name
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def plot_gray(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(to_uint8(image), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(to_uint8(image), cv2.COLOR_BGR2RGB))
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
    if float(image_float.max()) <= 1.0 and float(image_float.min()) >= 0.0:
        scaled = image_float * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    return normalize_channel(image_float)


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), to_uint8(image))
    if not success:
        raise OSError(f"Could not save image: {path}")
    print(f"Saved image: {path}")
    return path
