from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common import COUNTING_METADATA_PATH, DATASET_SEED, IMAGE_SIZE, SYNTHETIC_COUNT_DIR, build_background, ensure_base_directories, palette_bgr, save_image, write_csv


@dataclass
class SceneConfig:
    sample_id: str
    count: int
    overlap_level: str
    touching_pairs: int
    distractor_count: int
    occluder_count: int
    radius_range: tuple[int, int]
    texture_strength: int


def build_scene_configs() -> list[SceneConfig]:
    configs: list[SceneConfig] = []
    overlap_levels = ["low", "medium", "high", "extreme"]
    for index in range(96):
        difficulty_index = index // 24
        overlap_level = overlap_levels[min(difficulty_index, len(overlap_levels) - 1)]
        base_count = 6 + (index % 24)
        count = min(base_count, 25)
        touching_pairs = min(1 + difficulty_index + (index % 4), max(2, count // 3))
        distractor_count = 10 + difficulty_index * 6 + (index % 5)
        occluder_count = difficulty_index + (1 if index % 6 == 0 else 0)
        density_penalty = max(0, count - 12)
        min_radius = max(16, 40 - difficulty_index * 4 - density_penalty // 2)
        max_radius = max(min_radius + 4, 50 - difficulty_index * 3 - density_penalty)
        texture_strength = 6 + difficulty_index * 3
        configs.append(
            SceneConfig(
                sample_id=f"count_{index + 1:03d}",
                count=count,
                overlap_level=overlap_level,
                touching_pairs=touching_pairs,
                distractor_count=distractor_count,
                occluder_count=occluder_count,
                radius_range=(min_radius, max_radius),
                texture_strength=texture_strength,
            )
        )
    return configs


def place_circles(config: SceneConfig, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    height, width = IMAGE_SIZE
    circles: list[tuple[int, int, int]] = []
    margin = 56
    min_gap = 8

    def respects_clearance(center_x: int, center_y: int, radius: int, ignore_indices: set[int] | None = None) -> bool:
        ignore_indices = ignore_indices or set()
        for circle_index, (other_x, other_y, other_r) in enumerate(circles):
            if circle_index in ignore_indices:
                continue
            distance = float(np.hypot(center_x - other_x, center_y - other_y))
            if distance < radius + other_r + min_gap:
                return False
        return True

    for index in range(config.count):
        radius = int(rng.integers(config.radius_range[0], config.radius_range[1] + 1))
        placed = False
        center_x = 0
        center_y = 0

        if 0 < index <= config.touching_pairs:
            anchor_x, anchor_y, anchor_r = circles[index - 1]
            for _ in range(300):
                offset_angle = float(rng.uniform(0.0, 2.0 * np.pi))
                overlap_depth = int(rng.integers(6, 15))
                offset_distance = max(radius + anchor_r - overlap_depth, 10)
                candidate_x = int(np.clip(anchor_x + np.cos(offset_angle) * offset_distance, margin, width - margin))
                candidate_y = int(np.clip(anchor_y + np.sin(offset_angle) * offset_distance, margin, height - margin))
                if respects_clearance(candidate_x, candidate_y, radius, ignore_indices={index - 1}):
                    center_x, center_y = candidate_x, candidate_y
                    placed = True
                    break

        if not placed:
            for _ in range(500):
                candidate_x = int(rng.integers(margin, width - margin))
                candidate_y = int(rng.integers(margin, height - margin))
                if respects_clearance(candidate_x, candidate_y, radius):
                    center_x, center_y = candidate_x, candidate_y
                    placed = True
                    break

        if not placed:
            raise RuntimeError(f"Could not place circle for {config.sample_id}")
        circles.append((center_x, center_y, radius))

    return circles


def draw_textured_circle(image: np.ndarray, center_x: int, center_y: int, radius: int, color: tuple[int, int, int], rng: np.random.Generator, texture_strength: int) -> None:
    cv2.circle(image, (center_x, center_y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, (center_x, center_y), radius, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(image, (center_x - radius // 3, center_y - radius // 3), max(6, radius // 4), tuple(min(255, channel + 24) for channel in color), -1, cv2.LINE_AA)
    for _ in range(texture_strength):
        local_radius = int(rng.integers(max(4, radius // 8), max(5, radius // 3)))
        local_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        local_distance = float(rng.uniform(0.0, radius * 0.45))
        local_x = int(round(center_x + np.cos(local_angle) * local_distance))
        local_y = int(round(center_y + np.sin(local_angle) * local_distance))
        tint = tuple(max(0, min(255, channel + int(rng.integers(-20, 21)))) for channel in color)
        cv2.circle(image, (local_x, local_y), local_radius, tint, -1, cv2.LINE_AA)


def draw_small_distractors(image: np.ndarray, rng: np.random.Generator, count: int) -> None:
    colors = palette_bgr()
    height, width = image.shape[:2]
    for index in range(count):
        color = colors[index % len(colors)]
        center = (int(rng.integers(40, width - 40)), int(rng.integers(40, height - 40)))
        radius = int(rng.integers(4, 9))
        if index % 2 == 0:
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        else:
            axis_x = int(rng.integers(6, 12))
            axis_y = int(rng.integers(4, 9))
            angle = float(rng.uniform(0.0, 180.0))
            cv2.ellipse(image, center, (axis_x, axis_y), angle, 0.0, 360.0, color, -1, cv2.LINE_AA)


def draw_low_saturation_occluders(image: np.ndarray, rng: np.random.Generator, count: int) -> None:
    height, width = image.shape[:2]
    for _ in range(count):
        x0 = int(rng.integers(60, width - 200))
        y0 = int(rng.integers(60, height - 200))
        x1 = x0 + int(rng.integers(70, 180))
        y1 = y0 + int(rng.integers(18, 44))
        color_value = int(rng.integers(120, 180))
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (color_value, color_value, color_value), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0, image)


def render_scene(config: SceneConfig, rng: np.random.Generator) -> np.ndarray:
    background = build_background(IMAGE_SIZE, rng)
    circles = place_circles(config, rng)
    colors = palette_bgr()
    image = background.copy()

    for index, (center_x, center_y, radius) in enumerate(circles):
        base_color = colors[index % len(colors)]
        jittered = tuple(max(0, min(255, channel + int(rng.integers(-14, 15)))) for channel in base_color)
        draw_textured_circle(image, center_x, center_y, radius, jittered, rng, config.texture_strength)

    draw_small_distractors(image, rng, config.distractor_count)
    draw_low_saturation_occluders(image, rng, config.occluder_count)
    return image


def main() -> None:
    ensure_base_directories()
    rng = np.random.default_rng(DATASET_SEED)

    rows: list[dict[str, object]] = []
    for config in build_scene_configs():
        image = render_scene(config, rng)
        filename = f"{config.sample_id}.png"
        save_image(SYNTHETIC_COUNT_DIR / filename, image)
        rows.append(
            {
                "sample_id": config.sample_id,
                "filename": filename,
                "true_count": config.count,
                "overlap_level": config.overlap_level,
                "notes": (
                    f"touching_pairs={config.touching_pairs}; "
                    f"distractors={config.distractor_count}; "
                    f"occluders={config.occluder_count}; "
                    f"texture_strength={config.texture_strength}"
                ),
                "split": "synthetic",
                "source_dataset": "synthetic_counting",
            }
        )

    write_csv(
        COUNTING_METADATA_PATH,
        rows,
        fieldnames=["sample_id", "filename", "true_count", "overlap_level", "notes", "split", "source_dataset"],
    )
    print(f"Generated {len(rows)} synthetic counting samples in {SYNTHETIC_COUNT_DIR}")


if __name__ == "__main__":
    main()
