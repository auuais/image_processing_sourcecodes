from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_show_flag(description: str) -> bool:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated Matplotlib window after saving the output.",
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


def list_available_images(color_only: bool = False, grayscale_only: bool = False) -> list[str]:
    names: list[str] = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        sample = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if sample is None:
            continue

        is_color = sample.ndim == 3 and sample.shape[2] >= 3
        if color_only and not is_color:
            continue
        if grayscale_only and is_color:
            continue

        names.append(path.name)

    return names


def choose_image_name(
    image_name: str | None = None,
    *,
    color_only: bool = False,
    grayscale_only: bool = False,
    prompt: str = "Select an image",
) -> str:
    candidates = list_available_images(color_only=color_only, grayscale_only=grayscale_only)
    if not candidates:
        raise FileNotFoundError(f"No matching images found in {DATA_DIR}")

    if image_name is not None:
        if image_name not in candidates:
            available = ", ".join(candidates)
            raise ValueError(f"Unknown image '{image_name}'. Available choices: {available}")
        return image_name

    print(prompt)
    for index, candidate in enumerate(candidates, start=1):
        print(f"  {index}. {candidate}")

    while True:
        choice = input(f"Choose image [1-{len(candidates)}] (Enter=1): ").strip()
        if not choice:
            return candidates[0]
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        if choice in candidates:
            return choice
        print("Invalid selection. Enter a number from the list or an exact file name.")


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(to_uint8(image), cv2.COLOR_BGR2RGB))
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
