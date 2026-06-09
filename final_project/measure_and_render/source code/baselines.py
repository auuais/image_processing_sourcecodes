from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common import palette_bgr, save_image


def render_numbered_points(
    image: np.ndarray,
    points: list[tuple[int, int]],
    labels: list[str] | None = None,
    radius: int = 16,
    prefix_text: str | None = None,
) -> np.ndarray:
    overlay = image.copy()
    colors = palette_bgr()
    for index, point in enumerate(points):
        label = labels[index] if labels is not None else str(index + 1)
        color = colors[index % len(colors)]
        cv2.circle(overlay, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, point, radius, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (point[0] - 8, point[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if prefix_text:
        cv2.putText(overlay, prefix_text, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return overlay


def render_counting_som_overlay(image: np.ndarray, centroids: list[tuple[int, int]]) -> np.ndarray:
    return render_numbered_points(image, centroids, radius=15)


def render_angle_som_overlay(
    image: np.ndarray,
    vertex: tuple[int, int],
    endpoint_a: tuple[int, int],
    endpoint_b: tuple[int, int],
) -> np.ndarray:
    overlay = image.copy()
    cv2.line(overlay, vertex, endpoint_a, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.line(overlay, vertex, endpoint_b, (80, 80, 80), 2, cv2.LINE_AA)
    return render_numbered_points(overlay, [vertex, endpoint_a, endpoint_b], labels=["1", "2", "3"], radius=15)


def save_variant_image(variants_dir: Path, condition: str, sample_id: str, image: np.ndarray, suffix: str | None = None) -> Path:
    filename = f"{sample_id}.png" if suffix is None else f"{sample_id}_{suffix}.png"
    return save_image(variants_dir / condition / filename, image)


def format_count_text(value: int) -> str:
    noun = "object" if value == 1 else "objects"
    return f"A classical CV tool estimated {value} foreground {noun}."


def format_angle_text(value: float) -> str:
    return f"A classical CV tool estimated the angle as {value:.1f} degrees."


def counting_prompt(condition: str, measurement_text: str | None = None) -> str:
    prompt_base = "How many distinct foreground objects are present in this image? Answer with a single integer."
    if condition == "cot":
        return f"{prompt_base} Think step by step, then place the final integer on the last line."
    if condition == "grid":
        return "How many distinct foreground objects are present in this grid-labeled image? Answer with a single integer."
    if condition == "som":
        return "How many distinct foreground objects are marked by the numbered reference dots? Answer with a single integer."
    if condition == "pixels":
        return "How many distinct foreground objects are present? Use the overlaid numbered markers and any numeric cues in the image. Answer with a single integer."
    if condition == "text":
        return f"{prompt_base} {measurement_text}"
    if condition == "both":
        return f"{prompt_base} {measurement_text}"
    return prompt_base


def angle_prompt(condition: str, measurement_text: str | None = None) -> str:
    prompt_base = "Estimate the angle in degrees between the two main rays in the image. Answer with a single number."
    if condition == "cot":
        return f"{prompt_base} Think step by step, then place the final number on the last line."
    if condition == "grid":
        return "Estimate the angle in degrees between the two main rays in this grid-labeled image. Answer with a single number."
    if condition == "som":
        return "Estimate the angle in degrees at marker 1 between the segments pointing toward markers 2 and 3. Answer with a single number."
    if condition == "pixels":
        return "Estimate the angle in degrees using the overlaid rays, arc, and any numeric cues in the image. Answer with a single number."
    if condition == "text":
        return f"{prompt_base} {measurement_text}"
    if condition == "both":
        return f"{prompt_base} {measurement_text}"
    return prompt_base
