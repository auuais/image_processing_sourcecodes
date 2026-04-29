from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from common import DATA_DIR, REFERENCE_DIR, SAMPLES_DIR, clear_directory, ensure_color, ensure_dirs, save_image, write_csv


METADATA_NAME = "sample_metadata.csv"
CANVAS_WIDTH = 440
CANVAS_HEIGHT = 620
BG_COLOR = (242, 244, 246)


@dataclass(frozen=True)
class LabelSource:
    slug: str
    label_name: str
    image_url: str
    product_url: str
    primary_defect: str
    secondary_defect: str


@dataclass(frozen=True)
class SampleTemplate:
    suffix: str
    expected_status: str
    defect_role: str
    angle: float
    scale: float
    tx: float
    ty: float
    alpha: float = 1.0
    beta: int = 0


LABEL_SOURCES = [
    LabelSource(
        slug="king_cookies",
        label_name="KING Cookies",
        image_url="https://images.openfoodfacts.net/images/products/611/125/934/3108/front_fr.25.400.jpg",
        product_url="https://world.openfoodfacts.org/product/6111259343108",
        primary_defect="missing_print",
        secondary_defect="stain",
    ),
    LabelSource(
        slug="weetabix",
        label_name="Weetabix",
        image_url="https://images.openfoodfacts.net/images/products/501/002/900/0016/front_en.44.400.jpg",
        product_url="https://world.openfoodfacts.org/product/5010029000016",
        primary_defect="scratch",
        secondary_defect="corner_damage",
    ),
    LabelSource(
        slug="pringles_original",
        label_name="Pringles Original",
        image_url="https://images.openfoodfacts.net/images/products/505/399/010/1573/front_en.44.400.jpg",
        product_url="https://world.openfoodfacts.org/product/5053990101573",
        primary_defect="missing_print",
        secondary_defect="scratch",
    ),
    LabelSource(
        slug="green_tea",
        label_name="Jasmine Green Tea",
        image_url="https://images.openfoodfacts.net/images/products/692/381/881/2082/front_en.5.400.jpg",
        product_url="https://world.openfoodfacts.org/product/6923818812082",
        primary_defect="stain",
        secondary_defect="corner_damage",
    ),
    LabelSource(
        slug="lindt_90",
        label_name="Lindt 90% Cocoa",
        image_url="https://images.openfoodfacts.net/images/products/304/692/002/9759/front_en.492.400.jpg",
        product_url="https://world.openfoodfacts.org/product/3046920029759",
        primary_defect="missing_print",
        secondary_defect="scratch",
    ),
]


SAMPLE_TEMPLATES = [
    SampleTemplate("ok_pose_a", "PASS", "none", angle=7.0, scale=0.97, tx=10.0, ty=10.0, alpha=1.01, beta=1),
    SampleTemplate("ok_pose_b", "PASS", "none", angle=-8.0, scale=1.02, tx=-12.0, ty=8.0, alpha=0.99, beta=-2),
    SampleTemplate("fail_primary", "FAIL", "primary", angle=6.0, scale=0.96, tx=12.0, ty=14.0),
    SampleTemplate("fail_secondary", "FAIL", "secondary", angle=-10.0, scale=0.95, tx=-8.0, ty=14.0),
]


DEFECT_DESCRIPTIONS = {
    "none": "Normal package front with pose variation only.",
    "missing_print": "Missing print / erased label region.",
    "scratch": "Linear scratch across the front label.",
    "stain": "Surface contamination or stain on the package front.",
    "corner_damage": "Corner tear or crushed package edge.",
}


def download_color_image(url: str) -> np.ndarray:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode downloaded image: {url}")
    return image


def prepare_reference_canvas(image: np.ndarray) -> np.ndarray:
    image = ensure_color(image)
    height, width = image.shape[:2]
    scale = min((CANVAS_WIDTH - 64) / width, (CANVAS_HEIGHT - 64) / height)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    gradient = np.linspace(-5.0, 5.0, CANVAS_HEIGHT, dtype=np.float32).reshape(CANVAS_HEIGHT, 1, 1)
    canvas = np.clip(canvas.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    y0 = (CANVAS_HEIGHT - resized.shape[0]) // 2
    x0 = (CANVAS_WIDTH - resized.shape[1]) // 2
    shadow = canvas.copy()
    cv2.rectangle(shadow, (x0 + 8, y0 + 8), (x0 + resized.shape[1] + 8, y0 + resized.shape[0] + 8), (220, 222, 225), -1)
    shadow = cv2.GaussianBlur(shadow, (21, 21), 0)
    canvas = cv2.addWeighted(shadow, 0.18, canvas, 0.82, 0)
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (16, 16), (CANVAS_WIDTH - 17, CANVAS_HEIGHT - 17), (220, 224, 228), 2)
    return canvas


def affine_transform(image: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BG_COLOR,
    )


def apply_camera_variation(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_scratch(image: np.ndarray) -> np.ndarray:
    damaged = image.copy()
    height, width = damaged.shape[:2]
    start = (int(width * 0.18), int(height * 0.22))
    end = (int(width * 0.82), int(height * 0.74))
    cv2.line(damaged, start, end, (40, 40, 40), 5, cv2.LINE_AA)
    cv2.line(damaged, (start[0] + 4, start[1] - 2), (end[0] + 4, end[1] - 2), (225, 225, 225), 1, cv2.LINE_AA)
    return damaged


def add_missing_print(image: np.ndarray) -> np.ndarray:
    damaged = image.copy()
    height, width = damaged.shape[:2]
    x0, y0 = int(width * 0.24), int(height * 0.28)
    x1, y1 = int(width * 0.52), int(height * 0.58)
    roi = damaged[y0:y1, x0:x1]
    block = np.full_like(roi, 236)
    cv2.GaussianBlur(block, (9, 9), 0, dst=block)
    damaged[y0:y1, x0:x1] = block
    return damaged


def add_stain(image: np.ndarray) -> np.ndarray:
    damaged = image.copy()
    height, width = damaged.shape[:2]
    center = (int(width * 0.68), int(height * 0.52))
    axes = (int(width * 0.11), int(height * 0.08))
    cv2.ellipse(damaged, center, axes, -18, 0, 360, (55, 55, 55), -1, cv2.LINE_AA)
    cv2.ellipse(damaged, center, (axes[0] // 2, axes[1] // 2), 16, 0, 360, (125, 125, 125), -1, cv2.LINE_AA)
    return damaged


def add_corner_damage(image: np.ndarray) -> np.ndarray:
    damaged = image.copy()
    height, width = damaged.shape[:2]
    points = np.array(
        [
            [int(width * 0.70), int(height * 0.11)],
            [int(width * 0.88), int(height * 0.15)],
            [int(width * 0.82), int(height * 0.30)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(damaged, points, BG_COLOR, cv2.LINE_AA)
    cv2.polylines(damaged, [points], isClosed=True, color=(120, 120, 120), thickness=2, lineType=cv2.LINE_AA)
    return damaged


MODIFIERS = {
    "missing_print": add_missing_print,
    "scratch": add_scratch,
    "stain": add_stain,
    "corner_damage": add_corner_damage,
}


def resolve_defect(label: LabelSource, role: str) -> str:
    if role == "primary":
        return label.primary_defect
    if role == "secondary":
        return label.secondary_defect
    return "none"


def sample_description(label_name: str, expected_status: str, defect_name: str) -> str:
    if expected_status == "PASS":
        return f"{label_name}: {DEFECT_DESCRIPTIONS['none']}"
    return f"{label_name}: {DEFECT_DESCRIPTIONS[defect_name]}"


def generate_dataset() -> Path:
    ensure_dirs()
    clear_directory(REFERENCE_DIR)
    clear_directory(SAMPLES_DIR)

    rows: list[dict[str, object]] = []
    for label in LABEL_SOURCES:
        reference_raw = download_color_image(label.image_url)
        reference_image = prepare_reference_canvas(reference_raw)
        reference_filename = f"{label.slug}_reference.png"
        save_image(REFERENCE_DIR / reference_filename, reference_image)

        for template in SAMPLE_TEMPLATES:
            transformed = affine_transform(reference_image, template.angle, template.scale, template.tx, template.ty)
            transformed = apply_camera_variation(transformed, template.alpha, template.beta)
            defect_name = resolve_defect(label, template.defect_role)
            if defect_name != "none":
                transformed = MODIFIERS[defect_name](transformed)

            sample_filename = f"{label.slug}__{template.suffix}.png"
            save_image(SAMPLES_DIR / sample_filename, transformed)
            rows.append(
                {
                    "label_slug": label.slug,
                    "label_name": label.label_name,
                    "reference_filename": reference_filename,
                    "source_url": label.image_url,
                    "product_url": label.product_url,
                    "filename": sample_filename,
                    "expected_status": template.expected_status,
                    "defect_type": defect_name,
                    "description": sample_description(label.label_name, template.expected_status, defect_name),
                }
            )

    metadata_path = DATA_DIR / METADATA_NAME
    write_csv(
        metadata_path,
        ["label_slug", "label_name", "reference_filename", "source_url", "product_url", "filename", "expected_status", "defect_type", "description"],
        rows,
    )
    return metadata_path


if __name__ == "__main__":
    generate_dataset()
