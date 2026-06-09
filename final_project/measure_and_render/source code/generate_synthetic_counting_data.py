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
    shape_radius_range: tuple[int, int]


SCENE_CONFIGS = [
    SceneConfig("count_01", 6, "low", 0, (34, 48)),
    SceneConfig("count_02", 7, "low", 1, (32, 46)),
    SceneConfig("count_03", 8, "medium", 2, (32, 48)),
    SceneConfig("count_04", 9, "medium", 2, (30, 44)),
    SceneConfig("count_05", 10, "medium", 3, (30, 42)),
    SceneConfig("count_06", 11, "medium", 3, (28, 40)),
    SceneConfig("count_07", 8, "high", 3, (34, 50)),
    SceneConfig("count_08", 9, "high", 4, (34, 48)),
    SceneConfig("count_09", 10, "high", 4, (32, 46)),
    SceneConfig("count_10", 12, "high", 5, (28, 40)),
    SceneConfig("count_11", 13, "very_high", 6, (28, 38)),
    SceneConfig("count_12", 14, "very_high", 6, (26, 36)),
]


def place_circles(config: SceneConfig, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    height, width = IMAGE_SIZE
    circles: list[tuple[int, int, int]] = []
    margin = 80
    min_gap = 18

    def respects_clearance(center_x: int, center_y: int, radius: int, ignore_index: int | None = None) -> bool:
        for circle_index, (other_x, other_y, other_r) in enumerate(circles):
            if ignore_index is not None and circle_index == ignore_index:
                continue
            distance = float(np.hypot(center_x - other_x, center_y - other_y))
            if distance < radius + other_r + min_gap:
                return False
        return True

    for index in range(config.count):
        radius = int(rng.integers(config.shape_radius_range[0], config.shape_radius_range[1] + 1))
        center_x = 0
        center_y = 0
        placed = False

        if 0 < index <= config.touching_pairs:
            anchor_x, anchor_y, anchor_r = circles[index - 1]
            for _ in range(240):
                offset_angle = float(rng.uniform(0, 2 * np.pi))
                overlap_depth = int(rng.integers(8, 14))
                offset_distance = max(radius + anchor_r - overlap_depth, 8)
                candidate_x = int(np.clip(anchor_x + np.cos(offset_angle) * offset_distance, margin, width - margin))
                candidate_y = int(np.clip(anchor_y + np.sin(offset_angle) * offset_distance, margin, height - margin))
                if respects_clearance(candidate_x, candidate_y, radius, ignore_index=index - 1):
                    center_x, center_y = candidate_x, candidate_y
                    placed = True
                    break

        if not placed:
            for _ in range(300):
                candidate_x = int(rng.integers(margin, width - margin))
                candidate_y = int(rng.integers(margin, height - margin))
                if respects_clearance(candidate_x, candidate_y, radius):
                    center_x, center_y = candidate_x, candidate_y
                    placed = True
                    break

        if not placed:
            raise RuntimeError(f"Could not place circle for {config.sample_id} with count={config.count}")
        circles.append((center_x, center_y, radius))

    return np.zeros((height, width), dtype=np.uint8), circles


def render_scene(config: SceneConfig, rng: np.random.Generator) -> np.ndarray:
    background = build_background(IMAGE_SIZE, rng)
    _, circles = place_circles(config, rng)
    colors = palette_bgr()

    image = background.copy()
    for index, (center_x, center_y, radius) in enumerate(circles):
        color = colors[index % len(colors)]
        cv2.circle(image, (center_x, center_y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (center_x, center_y), radius, (40, 40, 40), 2, cv2.LINE_AA)
        # Small highlight to make the synthetic scenes less flat while keeping thresholding easy.
        cv2.circle(image, (center_x - radius // 3, center_y - radius // 3), max(6, radius // 4), tuple(min(255, c + 25) for c in color), -1, cv2.LINE_AA)
    return image


def main() -> None:
    ensure_base_directories()
    rng = np.random.default_rng(DATASET_SEED)

    rows: list[dict[str, object]] = []
    for config in SCENE_CONFIGS:
        image = render_scene(config, rng)
        filename = f"{config.sample_id}.png"
        save_image(SYNTHETIC_COUNT_DIR / filename, image)
        rows.append(
            {
                "sample_id": config.sample_id,
                "filename": filename,
                "true_count": config.count,
                "overlap_level": config.overlap_level,
                "notes": f"touching_pairs={config.touching_pairs}",
            }
        )

    write_csv(
        COUNTING_METADATA_PATH,
        rows,
        fieldnames=["sample_id", "filename", "true_count", "overlap_level", "notes"],
    )
    print(f"Generated {len(rows)} synthetic counting samples in {SYNTHETIC_COUNT_DIR}")


if __name__ == "__main__":
    main()
