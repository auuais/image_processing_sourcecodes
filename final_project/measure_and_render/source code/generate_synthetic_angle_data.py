from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common import ANGLE_METADATA_PATH, DATASET_SEED, IMAGE_SIZE, SYNTHETIC_ANGLE_DIR, build_background, ensure_base_directories, palette_bgr, save_image, write_csv


@dataclass
class AngleSceneConfig:
    sample_id: str
    true_angle_deg: float
    clutter_level: str
    clutter_segments: int
    nuisance_shapes: int


SCENE_CONFIGS = [
    AngleSceneConfig("angle_01", 24.0, "low", 0, 1),
    AngleSceneConfig("angle_02", 32.0, "low", 1, 1),
    AngleSceneConfig("angle_03", 41.0, "low", 1, 2),
    AngleSceneConfig("angle_04", 55.0, "medium", 2, 2),
    AngleSceneConfig("angle_05", 68.0, "medium", 2, 2),
    AngleSceneConfig("angle_06", 82.0, "medium", 2, 3),
    AngleSceneConfig("angle_07", 97.0, "medium", 3, 3),
    AngleSceneConfig("angle_08", 109.0, "high", 3, 3),
    AngleSceneConfig("angle_09", 121.0, "high", 3, 4),
    AngleSceneConfig("angle_10", 136.0, "high", 4, 4),
    AngleSceneConfig("angle_11", 48.0, "very_high", 4, 5),
    AngleSceneConfig("angle_12", 73.0, "very_high", 5, 5),
]


def endpoint_from_polar(origin: tuple[int, int], angle_deg: float, length: float) -> tuple[int, int]:
    radians = np.deg2rad(angle_deg)
    x = int(round(origin[0] + np.cos(radians) * length))
    y = int(round(origin[1] + np.sin(radians) * length))
    return x, y


def draw_ray(image: np.ndarray, origin: tuple[int, int], angle_deg: float, length: float, color: tuple[int, int, int], thickness: int) -> None:
    endpoint = endpoint_from_polar(origin, angle_deg, length)
    cv2.line(image, origin, endpoint, color, thickness, cv2.LINE_AA)


def draw_nuisance_shapes(image: np.ndarray, rng: np.random.Generator, count: int) -> None:
    colors = palette_bgr()
    height, width = image.shape[:2]
    for index in range(count):
        color = tuple(int(0.45 * channel + 120) for channel in colors[index % len(colors)])
        if index % 2 == 0:
            center = (int(rng.integers(80, width - 80)), int(rng.integers(80, height - 80)))
            radius = int(rng.integers(18, 42))
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        else:
            x0 = int(rng.integers(70, width - 170))
            y0 = int(rng.integers(70, height - 170))
            x1 = x0 + int(rng.integers(40, 110))
            y1 = y0 + int(rng.integers(40, 110))
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)


def render_scene(config: AngleSceneConfig, rng: np.random.Generator) -> np.ndarray:
    image = build_background(IMAGE_SIZE, rng)
    height, width = IMAGE_SIZE
    origin = (
        width // 2 + int(rng.integers(-35, 36)),
        height // 2 + int(rng.integers(-35, 36)),
    )

    base_angle = float(rng.uniform(-150.0, 150.0))
    second_angle = base_angle + config.true_angle_deg
    ray_length_1 = float(rng.uniform(220, 280))
    ray_length_2 = float(rng.uniform(220, 280))

    primary_colors = [(35, 35, 35), (55, 55, 55)]
    draw_ray(image, origin, base_angle, ray_length_1, primary_colors[0], thickness=12)
    draw_ray(image, origin, second_angle, ray_length_2, primary_colors[1], thickness=12)
    cv2.circle(image, origin, 11, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.circle(image, origin, 11, (40, 40, 40), 2, cv2.LINE_AA)

    for clutter_index in range(config.clutter_segments):
        while True:
            line_origin = (
                int(rng.integers(50, width - 50)),
                int(rng.integers(50, height - 50)),
            )
            if np.hypot(line_origin[0] - origin[0], line_origin[1] - origin[1]) > 170:
                break
        clutter_angle = float(rng.uniform(-180, 180))
        clutter_length = float(rng.uniform(55, 120))
        clutter_color = (110 + 8 * clutter_index, 110 + 8 * clutter_index, 110 + 8 * clutter_index)
        draw_ray(image, line_origin, clutter_angle, clutter_length, clutter_color, thickness=5)

    draw_nuisance_shapes(image, rng, config.nuisance_shapes)
    return image


def main() -> None:
    ensure_base_directories()
    rng = np.random.default_rng(DATASET_SEED + 101)

    rows: list[dict[str, object]] = []
    for config in SCENE_CONFIGS:
        image = render_scene(config, rng)
        filename = f"{config.sample_id}.png"
        save_image(SYNTHETIC_ANGLE_DIR / filename, image)
        rows.append(
            {
                "sample_id": config.sample_id,
                "filename": filename,
                "true_angle_deg": f"{config.true_angle_deg:.1f}",
                "clutter_level": config.clutter_level,
                "notes": f"clutter_segments={config.clutter_segments}; nuisance_shapes={config.nuisance_shapes}",
            }
        )

    write_csv(
        ANGLE_METADATA_PATH,
        rows,
        fieldnames=["sample_id", "filename", "true_angle_deg", "clutter_level", "notes"],
    )
    print(f"Generated {len(rows)} synthetic angle samples in {SYNTHETIC_ANGLE_DIR}")


if __name__ == "__main__":
    main()
