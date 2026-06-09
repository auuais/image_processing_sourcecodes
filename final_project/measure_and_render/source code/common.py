from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REAL_DATA_DIR = DATA_DIR / "real"
OUTPUT_DIR = ROOT_DIR / "output"
DOCS_DIR = ROOT_DIR / "docs"
MODELS_DIR = ROOT_DIR / "models"

SYNTHETIC_COUNT_DIR = DATA_DIR / "synthetic_counting"
SYNTHETIC_ANGLE_DIR = DATA_DIR / "synthetic_angle"
REAL_COUNTING_DIR = REAL_DATA_DIR / "counting"
REAL_MEASUREBENCH_DIR = REAL_DATA_DIR / "measurebench"

SUITE_OUTPUT_DIR = OUTPUT_DIR / "suite"
APPENDIX_OUTPUT_DIR = OUTPUT_DIR / "appendix"
VLM_OUTPUT_DIR = OUTPUT_DIR / "vlm"
COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "comparison"

COUNTING_METADATA_PATH = SYNTHETIC_COUNT_DIR / "metadata.csv"
ANGLE_METADATA_PATH = SYNTHETIC_ANGLE_DIR / "metadata.csv"
REAL_COUNTING_METADATA_PATH = REAL_COUNTING_DIR / "metadata.csv"
REAL_MEASUREBENCH_METADATA_PATH = REAL_MEASUREBENCH_DIR / "metadata.csv"

IMAGE_SIZE = (720, 720)
DATASET_SEED = 20260610
BASE_CONDITIONS = ("raw", "cot", "grid", "som", "pixels", "text", "both")
TRUST_DELTA_COUNTING = (-3, -1, 1, 3)
TRUST_DELTA_ANGLE = (-10.0, -5.0, 5.0, 10.0)


@dataclass
class CountingSample:
    sample_id: str
    image_path: Path
    true_count: int
    overlap_level: str
    notes: str
    split: str = "synthetic"
    source_dataset: str = "synthetic_counting"


@dataclass
class AngleSample:
    sample_id: str
    image_path: Path
    true_angle_deg: float
    clutter_level: str
    notes: str
    split: str = "synthetic"
    source_dataset: str = "synthetic_angle"


def task_output_paths(task_name: str) -> dict[str, Path]:
    task_dir = OUTPUT_DIR / task_name
    return {
        "task_dir": task_dir,
        "per_sample_dir": task_dir / "per_sample",
        "variants_dir": task_dir / "variants",
        "metrics_path": task_dir / "metrics.csv",
        "manifest_path": task_dir / "vlm_manifest.jsonl",
        "results_path": task_dir / "RESULTS.md",
        "summary_plot_path": task_dir / f"{task_name}_summary.png",
    }


def ensure_base_directories() -> None:
    for path in [
        DATA_DIR,
        REAL_DATA_DIR,
        OUTPUT_DIR,
        DOCS_DIR,
        MODELS_DIR,
        SYNTHETIC_COUNT_DIR,
        SYNTHETIC_ANGLE_DIR,
        REAL_COUNTING_DIR,
        REAL_MEASUREBENCH_DIR,
        SUITE_OUTPUT_DIR,
        APPENDIX_OUTPUT_DIR,
        VLM_OUTPUT_DIR,
        COMPARISON_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_task_output_dirs(task_name: str) -> dict[str, Path]:
    ensure_base_directories()
    paths = task_output_paths(task_name)
    for key in ["task_dir", "per_sample_dir", "variants_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    for condition in BASE_CONDITIONS:
        (paths["variants_dir"] / condition).mkdir(parents=True, exist_ok=True)
    (paths["variants_dir"] / "text_prompts").mkdir(parents=True, exist_ok=True)
    return paths


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
    return path


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(ensure_color(image), cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")


def plot_gray(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(to_uint8(image), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


def finalize_figure(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def palette_bgr() -> list[tuple[int, int, int]]:
    return [
        (60, 140, 240),
        (90, 200, 70),
        (210, 100, 210),
        (250, 190, 60),
        (70, 210, 220),
        (200, 120, 70),
        (130, 90, 220),
        (80, 175, 120),
    ]


def build_background(size: tuple[int, int], rng: np.random.Generator | None = None) -> np.ndarray:
    height, width = size
    y_gradient = np.linspace(240, 226, height, dtype=np.float32)[:, None]
    x_gradient = np.linspace(0, 8, width, dtype=np.float32)[None, :]
    base = y_gradient - x_gradient
    background = np.stack([base, base, base], axis=-1)
    if rng is None:
        rng = np.random.default_rng(DATASET_SEED)
    noise = rng.normal(loc=0.0, scale=2.0, size=background.shape)
    return np.clip(background + noise, 0, 255).astype(np.uint8)


def add_measurement_footer(
    image: np.ndarray,
    text: str,
    note: str = "This rendered measurement is the scaffold under study, not ground truth.",
) -> np.ndarray:
    footer_height = 94
    canvas = np.full((image.shape[0] + footer_height, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[: image.shape[0]] = image
    cv2.rectangle(canvas, (0, image.shape[0]), (image.shape[1], canvas.shape[0]), (245, 245, 245), -1)
    cv2.putText(canvas, text, (24, image.shape[0] + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, note, (24, image.shape[0] + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2, cv2.LINE_AA)
    return canvas


def render_coordinate_grid(image: np.ndarray, divisions: int = 6) -> np.ndarray:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    step_x = width // divisions
    step_y = height // divisions
    for grid_index in range(1, divisions):
        cv2.line(overlay, (grid_index * step_x, 0), (grid_index * step_x, height), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, grid_index * step_y), (width, grid_index * step_y), (120, 120, 120), 1, cv2.LINE_AA)
    for row in range(divisions):
        for col in range(divisions):
            label = f"{chr(65 + row)}{col + 1}"
            anchor = (col * step_x + 12, row * step_y + 28)
            cv2.putText(overlay, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    return overlay


def slugify_delta(value: float) -> str:
    rounded = int(round(value)) if abs(value - round(value)) < 1e-6 else value
    sign = "p" if float(rounded) >= 0.0 else "m"
    magnitude = str(abs(rounded)).replace(".", "_")
    return f"{sign}{magnitude}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_counting_metadata(
    path: Path = COUNTING_METADATA_PATH,
    image_dir: Path = SYNTHETIC_COUNT_DIR,
    split: str = "synthetic",
    source_dataset: str = "synthetic_counting",
) -> list[CountingSample]:
    if not path.exists():
        raise FileNotFoundError(f"Counting metadata does not exist: {path}")
    samples: list[CountingSample] = []
    for row in read_csv_rows(path):
        samples.append(
            CountingSample(
                sample_id=row["sample_id"],
                image_path=image_dir / row["filename"],
                true_count=int(row["true_count"]),
                overlap_level=row.get("overlap_level", row.get("difficulty", "unknown")),
                notes=row.get("notes", ""),
                split=row.get("split", split),
                source_dataset=row.get("source_dataset", source_dataset),
            )
        )
    return samples


def read_angle_metadata(
    path: Path = ANGLE_METADATA_PATH,
    image_dir: Path = SYNTHETIC_ANGLE_DIR,
    split: str = "synthetic",
    source_dataset: str = "synthetic_angle",
) -> list[AngleSample]:
    if not path.exists():
        raise FileNotFoundError(f"Angle metadata does not exist: {path}")
    samples: list[AngleSample] = []
    for row in read_csv_rows(path):
        samples.append(
            AngleSample(
                sample_id=row["sample_id"],
                image_path=image_dir / row["filename"],
                true_angle_deg=float(row["true_angle_deg"]),
                clutter_level=row.get("clutter_level", row.get("difficulty", "unknown")),
                notes=row.get("notes", ""),
                split=row.get("split", split),
                source_dataset=row.get("source_dataset", source_dataset),
            )
        )
    return samples
