from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import ALIGNED_DIR, MASKS_DIR, MATCHES_DIR, OUTPUT_DIR, REFERENCE_DIR, REPORT_MD, RESULTS_CSV, SAMPLES_DIR, SUMMARIES_DIR, clear_directory, ensure_color, ensure_dirs, finalize_figure, load_color_image, plot_bgr, plot_gray, read_csv, save_image, write_csv
from generate_example_data import METADATA_NAME, generate_dataset


RATIO_TEST = 0.78
MIN_MATCHES = 12
DIFF_THRESHOLD = 34
MIN_REGION_AREA = 120
FAIL_AREA_RATIO = 0.0070
MAX_DRAW_MATCHES = 40
LABEL_SHORT_NAMES = {
    "king_cookies": "Cookies",
    "weetabix": "Weetabix",
    "pringles_original": "Pringles",
    "green_tea": "Green Tea",
    "lindt_90": "Lindt 90%",
}


def build_reference_cache(metadata: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for record in metadata:
        slug = record["label_slug"]
        if slug in cache:
            continue
        reference_color = load_color_image(REFERENCE_DIR / record["reference_filename"])
        cache[slug] = {
            "label_name": record["label_name"],
            "source_url": record["source_url"],
            "product_url": record["product_url"],
            "reference_color": reference_color,
            "reference_gray": cv2.cvtColor(reference_color, cv2.COLOR_BGR2GRAY),
        }
    return cache


def align_to_reference(reference_gray: np.ndarray, sample_gray: np.ndarray, sample_color: np.ndarray) -> dict[str, object]:
    sift = cv2.SIFT_create(nfeatures=900)
    kp_ref, desc_ref = sift.detectAndCompute(reference_gray, None)
    kp_sample, desc_sample = sift.detectAndCompute(sample_gray, None)
    if desc_ref is None or desc_sample is None:
        raise RuntimeError("SIFT descriptors could not be computed.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good_matches = []
    for pair in matcher.knnMatch(desc_sample, desc_ref, k=2):
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < RATIO_TEST * second.distance:
            good_matches.append(first)
    if len(good_matches) < MIN_MATCHES:
        raise RuntimeError(f"Not enough good matches for alignment: {len(good_matches)}")

    sample_points = np.float32([kp_sample[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ref_points = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(sample_points, ref_points, cv2.RANSAC, 3.5)
    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed.")

    height, width = reference_gray.shape
    aligned_color = cv2.warpPerspective(sample_color, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(242, 244, 246))
    aligned_gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY)
    valid_mask = cv2.warpPerspective(np.full(sample_gray.shape, 255, dtype=np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_mask = cv2.erode(valid_mask, np.ones((9, 9), np.uint8), iterations=1)

    inliers = [good_matches[index] for index, flag in enumerate(inlier_mask.ravel()) if flag]
    match_view = cv2.drawMatches(ensure_color(reference_gray), kp_ref, sample_color, kp_sample, inliers[:MAX_DRAW_MATCHES], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return {
        "aligned_color": aligned_color,
        "aligned_gray": aligned_gray,
        "valid_mask": valid_mask,
        "good_matches": len(good_matches),
        "inliers": len(inliers),
        "match_view": match_view,
    }


def locate_defects(reference_gray: np.ndarray, aligned_gray: np.ndarray, valid_mask: np.ndarray) -> dict[str, object]:
    diff = cv2.absdiff(cv2.GaussianBlur(reference_gray, (5, 5), 0), cv2.GaussianBlur(aligned_gray, (5, 5), 0))
    diff[valid_mask == 0] = 0
    _, raw_mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    filtered = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
    mask = np.zeros_like(filtered)
    regions: list[tuple[int, int, int, int, int]] = []
    for label_id in range(1, component_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < MIN_REGION_AREA:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        mask[labels == label_id] = 255
        regions.append((x, y, w, h, area))

    valid_area = max(int(np.count_nonzero(valid_mask)), 1)
    defect_area = int(np.count_nonzero(mask))
    defect_ratio = defect_area / valid_area
    return {
        "diff": diff,
        "mask": mask,
        "regions": regions,
        "total_defect_area": defect_area,
        "defect_ratio": defect_ratio,
        "predicted_status": "FAIL" if defect_ratio >= FAIL_AREA_RATIO else "PASS",
    }


def create_overlay(aligned_color: np.ndarray, mask: np.ndarray, predicted_status: str, expected_status: str, label_name: str, defect_type: str) -> np.ndarray:
    overlay = aligned_color.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.55, red_layer, 0.45, 0), overlay)
    status_color = (0, 180, 0) if predicted_status == "PASS" else (0, 0, 255)
    for index, text in enumerate([label_name, f"Predicted: {predicted_status}", f"Expected: {expected_status}", f"Defect: {defect_type}"]):
        y = 24 + index * 22
        cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, status_color, 2, cv2.LINE_AA)
    return overlay


def inspect_one_sample(record: dict[str, str], reference_cache: dict[str, dict[str, object]]) -> dict[str, object]:
    sample_path = SAMPLES_DIR / record["filename"]
    sample_color = load_color_image(sample_path)
    sample_gray = cv2.cvtColor(sample_color, cv2.COLOR_BGR2GRAY)
    reference_record = reference_cache[record["label_slug"]]
    alignment = align_to_reference(reference_record["reference_gray"], sample_gray, sample_color)
    defect_result = locate_defects(reference_record["reference_gray"], alignment["aligned_gray"], alignment["valid_mask"])
    overlay = create_overlay(alignment["aligned_color"], defect_result["mask"], str(defect_result["predicted_status"]), record["expected_status"], record["label_name"], record["defect_type"])
    for x, y, w, h, _area in defect_result["regions"]:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    stem = sample_path.stem
    save_image(ALIGNED_DIR / f"{stem}_aligned.png", alignment["aligned_color"])
    save_image(MASKS_DIR / f"{stem}_mask.png", defect_result["mask"])
    save_image(MATCHES_DIR / f"{stem}_matches.png", alignment["match_view"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_bgr(axes[0, 0], sample_color, f"Sample - {record['label_name']}")
    plot_bgr(axes[0, 1], alignment["match_view"], f"SIFT matches ({alignment['inliers']} inliers)")
    plot_bgr(axes[0, 2], alignment["aligned_color"], "Aligned sample")
    plot_gray(axes[1, 0], defect_result["diff"], "Difference map")
    plot_gray(axes[1, 1], defect_result["mask"], "Detected defect mask")
    plot_bgr(axes[1, 2], overlay, f"{defect_result['predicted_status']} - {record['defect_type']}")
    summary_path = SUMMARIES_DIR / f"{stem}_summary.png"
    fig.tight_layout()
    fig.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {summary_path}")

    return {
        "label_slug": record["label_slug"],
        "label_name": record["label_name"],
        "reference_filename": record["reference_filename"],
        "filename": sample_path.name,
        "expected_status": record["expected_status"],
        "predicted_status": defect_result["predicted_status"],
        "correct": str(defect_result["predicted_status"] == record["expected_status"]),
        "defect_type": record["defect_type"],
        "good_matches": int(alignment["good_matches"]),
        "inliers": int(alignment["inliers"]),
        "region_count": len(defect_result["regions"]),
        "defect_area": int(defect_result["total_defect_area"]),
        "defect_ratio": round(float(defect_result["defect_ratio"]), 6),
        "source_url": record["source_url"],
        "product_url": record["product_url"],
    }


def create_reference_gallery(reference_cache: dict[str, dict[str, object]]) -> None:
    items = sorted(reference_cache.items(), key=lambda pair: pair[1]["label_name"])
    fig, axes = plt.subplots(1, len(items), figsize=(3.0 * len(items), 4.8))
    if len(items) == 1:
        axes = [axes]
    for ax, (_slug, record) in zip(axes, items):
        plot_bgr(ax, record["reference_color"], record["label_name"])
    finalize_figure(fig, "reference_gallery.png")


def write_report(rows: list[dict[str, object]], reference_cache: dict[str, dict[str, object]]) -> None:
    total = len(rows)
    correct = sum(row["correct"] == "True" for row in rows)
    accuracy = correct / total if total else 0.0
    per_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_label[str(row["label_slug"])].append(row)

    label_lines = [f"- `{label_rows[0]['label_name']}`: `{sum(row['correct'] == 'True' for row in label_rows)}/{len(label_rows)}` correct" for _slug, label_rows in sorted(per_label.items())]
    source_lines = [f"- `{record['label_name']}`: {record['product_url']}" for _slug, record in sorted(reference_cache.items())]
    table_rows = "".join(
        f"| {row['label_name']} | {row['filename']} | {row['expected_status']} | {row['predicted_status']} | {row['correct']} | {row['defect_type']} | {row['defect_ratio']} | {row['inliers']} |\n"
        for row in rows
    )
    report = (
        "# Packaging Label Defect Inspection Results\n\n"
        "- Dataset source: Open Food Facts package-front images\n"
        f"- Labels inspected: `{len(reference_cache)}`\n- Samples inspected: `{total}`\n- Correct predictions: `{correct}`\n- Accuracy: `{accuracy:.1%}`\n\n"
        "## Per-label summary\n\n" + "\n".join(label_lines) + "\n\n"
        "## Per-sample results\n\n| Label | Sample | Expected | Predicted | Correct | Defect Type | Defect Ratio | Inliers |\n|---|---|---|---|---|---|---:|---:|\n"
        + table_rows
        + "\n## Source labels\n\n" + "\n".join(source_lines)
        + "\n\n## Interpretation\n\n- PASS samples should align well and produce only small residual differences.\n- FAIL samples keep large residual regions after alignment, which are localized as defect masks.\n- The pipeline uses SIFT-based alignment plus absdiff, thresholding, morphology, and connected components.\n"
    )
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Saved report: {REPORT_MD}")


def create_overview_figure(rows: list[dict[str, object]]) -> None:
    display_names, ratios, colors = [], [], []
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"PASS": [], "FAIL": []})
    for row in rows:
        label_short = LABEL_SHORT_NAMES.get(str(row["label_slug"]), str(row["label_name"]))
        variant = Path(str(row["filename"])).stem.split("__", 1)[1].replace("ok_pose_", "ok").replace("fail_", "f-")
        ratio = float(row["defect_ratio"]) * 100.0
        display_names.append(f"{label_short}\n{variant}")
        ratios.append(ratio)
        colors.append("#2ca02c" if row["predicted_status"] == "PASS" else "#d62728")
        by_label[str(row["label_slug"])][str(row["expected_status"])].append(ratio)

    ordered_labels = [slug for slug in LABEL_SHORT_NAMES if slug in by_label]
    label_names = [LABEL_SHORT_NAMES[slug] for slug in ordered_labels]
    pass_means = [np.mean(by_label[slug]["PASS"]) if by_label[slug]["PASS"] else 0.0 for slug in ordered_labels]
    fail_means = [np.mean(by_label[slug]["FAIL"]) if by_label[slug]["FAIL"] else 0.0 for slug in ordered_labels]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes[0].bar(display_names, ratios, color=colors)
    axes[0].axhline(FAIL_AREA_RATIO * 100.0, color="black", linestyle="--", linewidth=1.5, label="Fail threshold")
    axes[0].set_ylabel("Defect area (%)")
    axes[0].set_title("Predicted defect area by sample")
    axes[0].legend()

    x = np.arange(len(label_names))
    width = 0.34
    axes[1].bar(x - width / 2, pass_means, width, color="#2ca02c", label="PASS mean")
    axes[1].bar(x + width / 2, fail_means, width, color="#d62728", label="FAIL mean")
    axes[1].axhline(FAIL_AREA_RATIO * 100.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_names)
    axes[1].set_ylabel("Mean defect area (%)")
    axes[1].set_title("Per-label separation between PASS and FAIL samples")
    axes[1].legend()
    finalize_figure(fig, "demo_overview.png")


def main() -> None:
    ensure_dirs()
    metadata_path = generate_dataset()
    clear_directory(OUTPUT_DIR)
    ensure_dirs()

    metadata = read_csv(metadata_path)
    reference_cache = build_reference_cache(metadata)
    rows = [inspect_one_sample(record, reference_cache) for record in metadata]
    write_csv(
        RESULTS_CSV,
        ["label_slug", "label_name", "reference_filename", "filename", "expected_status", "predicted_status", "correct", "defect_type", "good_matches", "inliers", "region_count", "defect_area", "defect_ratio", "source_url", "product_url"],
        rows,
    )
    write_report(rows, reference_cache)
    create_reference_gallery(reference_cache)
    create_overview_figure(rows)


if __name__ == "__main__":
    main()
