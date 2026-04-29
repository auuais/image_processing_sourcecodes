from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
LENA_PATH = DATA_DIR / "lena.png"


def parse_show_flag(description: str) -> bool:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated Matplotlib window after saving the output.",
    )
    return parser.parse_args().show


def load_lena_color() -> np.ndarray:
    image = cv2.imread(str(LENA_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {LENA_PATH}")
    return image


def load_lena_gray() -> np.ndarray:
    image = cv2.imread(str(LENA_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {LENA_PATH}")
    return image


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")


def plot_gray(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
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


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel_float = channel.astype(np.float32)
    min_value = float(channel_float.min())
    max_value = float(channel_float.max())

    if max_value - min_value < 1e-6:
        return np.zeros(channel.shape, dtype=np.uint8)

    normalized = (channel_float - min_value) * 255.0 / (max_value - min_value)
    return normalized.astype(np.uint8)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    if float(image.max()) <= 1.0:
        scaled = image * 255.0
    else:
        scaled = image

    return np.clip(scaled, 0, 255).astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), to_uint8(image))
    if not success:
        raise OSError(f"Could not save image: {path}")
    print(f"Saved image: {path}")
    return path
