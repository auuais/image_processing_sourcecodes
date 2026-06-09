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
    dark_distractors: int
    ray_thickness: int


def build_scene_configs() -> list[AngleSceneConfig]:
    configs: list[AngleSceneConfig] = []
    clutter_levels = ["low", "medium", "high", "extreme"]
    angles = np.linspace(18.0, 162.0, 96)
    for index, angle in enumerate(angles):
        difficulty_index = index // 24
        clutter_level = clutter_levels[min(difficulty_index, len(clutter_levels) - 1)]
        configs.append(
            AngleSceneConfig(
                sample_id=f"angle_{index + 1:03d}",
                true_angle_deg=float(round(angle + ((index % 4) - 1.5) * 1.2, 1)),
                clutter_level=clutter_level,
                clutter_segments=1 + difficulty_index + (index % 2),
                nuisance_shapes=2 + difficulty_index + (index % 3),
                dark_distractors=0 if difficulty_index < 2 else (1 if index % 3 == 0 else 0),
                ray_thickness=12 if difficulty_index < 2 else 11 if difficulty_index == 2 else 10,
            )
        )
    return configs


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
        color = tuple(int(0.45 * channel + 125) for channel in colors[index % len(colors)])
        if index % 3 == 0:
            center = (int(rng.integers(60, width - 60)), int(rng.integers(60, height - 60)))
            radius = int(rng.integers(18, 44))
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        elif index % 3 == 1:
            x0 = int(rng.integers(60, width - 190))
            y0 = int(rng.integers(60, height - 190))
            x1 = x0 + int(rng.integers(45, 125))
            y1 = y0 + int(rng.integers(45, 125))
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
        else:
            center = (int(rng.integers(60, width - 60)), int(rng.integers(60, height - 60)))
            axes = (int(rng.integers(16, 38)), int(rng.integers(10, 24)))
            angle = float(rng.uniform(0.0, 180.0))
            cv2.ellipse(image, center, axes, angle, 0.0, 360.0, color, -1, cv2.LINE_AA)


def draw_clutter_segments(image: np.ndarray, rng: np.random.Generator, count: int, vertex: tuple[int, int], dark_distractors: int) -> None:
    height, width = image.shape[:2]
    for clutter_index in range(count):
        while True:
            line_origin = (int(rng.integers(40, width - 40)), int(rng.integers(40, height - 40)))
            if np.hypot(line_origin[0] - vertex[0], line_origin[1] - vertex[1]) > 165:
                break
        clutter_angle = float(rng.uniform(-180.0, 180.0))
        clutter_length = float(rng.uniform(55.0, 135.0))
        color_value = 95 + clutter_index * 7
        draw_ray(image, line_origin, clutter_angle, clutter_length, (color_value, color_value, color_value), thickness=5)

    for _ in range(dark_distractors):
        while True:
            line_origin = (int(rng.integers(40, width - 40)), int(rng.integers(40, height - 40)))
            if np.hypot(line_origin[0] - vertex[0], line_origin[1] - vertex[1]) > 120:
                break
        clutter_angle = float(rng.uniform(-180.0, 180.0))
        clutter_length = float(rng.uniform(70.0, 130.0))
        draw_ray(image, line_origin, clutter_angle, clutter_length, (58, 58, 58), thickness=6)


def render_scene(config: AngleSceneConfig, rng: np.random.Generator) -> np.ndarray:
    image = build_background(IMAGE_SIZE, rng)
    height, width = IMAGE_SIZE
    origin = (
        width // 2 + int(rng.integers(-70, 71)),
        height // 2 + int(rng.integers(-70, 71)),
    )

    base_angle = float(rng.uniform(-155.0, 155.0))
    second_angle = base_angle + config.true_angle_deg
    ray_length_1 = float(rng.uniform(210.0, 300.0))
    ray_length_2 = float(rng.uniform(210.0, 300.0))

    primary_colors = [(34, 34, 34), (54, 54, 54)]
    draw_ray(image, origin, base_angle, ray_length_1, primary_colors[0], thickness=config.ray_thickness)
    draw_ray(image, origin, second_angle, ray_length_2, primary_colors[1], thickness=config.ray_thickness)
    cv2.circle(image, origin, 12, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.circle(image, origin, 12, (38, 38, 38), 2, cv2.LINE_AA)

    draw_clutter_segments(image, rng, config.clutter_segments, origin, config.dark_distractors)
    draw_nuisance_shapes(image, rng, config.nuisance_shapes)
    return image


def main() -> None:
    ensure_base_directories()
    rng = np.random.default_rng(DATASET_SEED + 101)

    rows: list[dict[str, object]] = []
    for config in build_scene_configs():
        image = render_scene(config, rng)
        filename = f"{config.sample_id}.png"
        save_image(SYNTHETIC_ANGLE_DIR / filename, image)
        rows.append(
            {
                "sample_id": config.sample_id,
                "filename": filename,
                "true_angle_deg": f"{config.true_angle_deg:.1f}",
                "clutter_level": config.clutter_level,
                "notes": (
                    f"clutter_segments={config.clutter_segments}; "
                    f"nuisance_shapes={config.nuisance_shapes}; "
                    f"dark_distractors={config.dark_distractors}; "
                    f"ray_thickness={config.ray_thickness}"
                ),
                "split": "synthetic",
                "source_dataset": "synthetic_angle",
            }
        )

    write_csv(
        ANGLE_METADATA_PATH,
        rows,
        fieldnames=["sample_id", "filename", "true_angle_deg", "clutter_level", "notes", "split", "source_dataset"],
    )
    print(f"Generated {len(rows)} synthetic angle samples in {SYNTHETIC_ANGLE_DIR}")


if __name__ == "__main__":
    main()
