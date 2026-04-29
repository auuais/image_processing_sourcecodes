from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import data


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
BOW_DIR = DATA_DIR / "bow"
TRAFFIC_VIDEO_PATH = DATA_DIR / "traffic.mp4"
SURF_AVAILABLE = hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create")
BRIEF_AVAILABLE = hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "BriefDescriptorExtractor_create")


def parse_show_flag(description: str) -> bool:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib window after saving the output.")
    return parser.parse_args().show


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_float = image.astype(np.float32)
    if float(image_float.min()) >= 0.0 and float(image_float.max()) <= 1.0:
        return np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
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
    print(f"Saved image: {path}")
    return path


def plot_bgr(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(ensure_color(image), cv2.COLOR_BGR2RGB))
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


def load_color_image(name: str) -> np.ndarray:
    ensure_demo_data()
    image = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {DATA_DIR / name}")
    return image


def load_gray_image(name: str) -> np.ndarray:
    ensure_demo_data()
    image = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {DATA_DIR / name}")
    return image


def surf_label() -> str:
    return "SURF" if SURF_AVAILABLE else "SIFT fallback for SURF"


def brief_label() -> str:
    return "BRIEF" if BRIEF_AVAILABLE else "BRISK fallback for BRIEF"


def resize_for_demo(image: np.ndarray, width: int = 640) -> np.ndarray:
    image = ensure_color(image)
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


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
        borderMode=cv2.BORDER_REFLECT101,
    )


def create_match_pair() -> None:
    ref_path = DATA_DIR / "match_scene_reference.png"
    query_path = DATA_DIR / "match_scene_query.png"
    if ref_path.exists() and query_path.exists():
        return

    reference = resize_for_demo(data.coffee(), width=720)
    query = affine_transform(reference, angle=8.0, scale=0.96, tx=18.0, ty=14.0)
    query = cv2.convertScaleAbs(query, alpha=1.03, beta=4)
    save_image(ref_path, reference)
    save_image(query_path, query)


def create_bow_dataset() -> None:
    classes = {
        "astronaut": resize_for_demo(data.astronaut(), width=360),
        "coffee": resize_for_demo(data.coffee(), width=360),
        "camera": resize_for_demo(data.camera(), width=360),
    }
    transforms = [
        {"angle": 0.0, "scale": 1.00, "tx": 0.0, "ty": 0.0, "alpha": 1.00, "beta": 0},
        {"angle": 5.0, "scale": 0.98, "tx": 8.0, "ty": 6.0, "alpha": 1.02, "beta": 2},
        {"angle": -6.0, "scale": 1.03, "tx": -10.0, "ty": 4.0, "alpha": 0.98, "beta": -4},
        {"angle": 9.0, "scale": 0.95, "tx": 10.0, "ty": 10.0, "alpha": 1.04, "beta": 3},
    ]

    for class_name, base_image in classes.items():
        class_dir = BOW_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index, spec in enumerate(transforms, start=1):
            out_path = class_dir / f"{class_name}_{index:02d}.png"
            if out_path.exists():
                continue
            augmented = affine_transform(base_image, spec["angle"], spec["scale"], spec["tx"], spec["ty"])
            augmented = cv2.convertScaleAbs(augmented, alpha=spec["alpha"], beta=spec["beta"])
            save_image(out_path, augmented)


def ensure_demo_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_match_pair()
    create_bow_dataset()


def traffic_video_exists() -> bool:
    return TRAFFIC_VIDEO_PATH.exists()


def load_video_frame_pair(
    video_path: Path,
    frame_a: int | None = None,
    frame_b: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    if frame_a is None:
        frame_a = min(30, max(0, frame_count // 12))
    if frame_b is None:
        default_gap = max(8, int(round(fps * 0.5))) if fps > 0 else 12
        frame_b = min(frame_count - 1, frame_a + default_gap)

    frame_a = max(0, min(frame_count - 1, int(frame_a)))
    frame_b = max(0, min(frame_count - 1, int(frame_b)))

    def read_frame(frame_index: int) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        return frame

    try:
        image_a = read_frame(frame_a)
        image_b = read_frame(frame_b)
    finally:
        capture.release()

    return image_a, image_b, {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "frame_count": frame_count,
        "fps": fps,
    }


def create_surf_detector():
    if SURF_AVAILABLE:
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    return cv2.SIFT_create(nfeatures=800)


def create_descriptor(name: str):
    name = name.upper()
    if name == "SURF":
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400) if SURF_AVAILABLE else cv2.SIFT_create(nfeatures=800)
    if name == "BRIEF":
        return cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32) if BRIEF_AVAILABLE else cv2.BRISK_create()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=1200)
    raise ValueError(f"Unsupported descriptor name: {name}")


def resolved_descriptor_label(name: str) -> str:
    if name.upper() == "SURF":
        return surf_label()
    if name.upper() == "BRIEF":
        return brief_label()
    return "ORB"


def descriptor_norm(name: str) -> int:
    if name.upper() == "SURF" and SURF_AVAILABLE:
        return cv2.NORM_L2
    if name.upper() == "SURF" and not SURF_AVAILABLE:
        return cv2.NORM_L2
    return cv2.NORM_HAMMING


def top_keypoints(keypoints: list[cv2.KeyPoint], max_points: int = 700) -> list[cv2.KeyPoint]:
    ranked = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    return ranked[:max_points]


def binary_ready_keypoints(keypoints: list[cv2.KeyPoint], min_size: float = 31.0) -> list[cv2.KeyPoint]:
    prepared = []
    for kp in keypoints:
        prepared.append(
            cv2.KeyPoint(
                x=float(kp.pt[0]),
                y=float(kp.pt[1]),
                size=max(float(kp.size), min_size),
                angle=float(kp.angle),
                response=float(kp.response),
                octave=int(kp.octave),
                class_id=int(kp.class_id),
            )
        )
    return prepared


def detect_surf_keypoints(gray: np.ndarray, max_points: int = 700) -> list[cv2.KeyPoint]:
    detector = create_surf_detector()
    keypoints = detector.detect(gray, None)
    return top_keypoints(list(keypoints), max_points=max_points)


def compute_descriptor(name: str, gray: np.ndarray, keypoints: list[cv2.KeyPoint]) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    descriptor_name = name.upper()
    prepared_keypoints = binary_ready_keypoints(keypoints) if descriptor_name in {"BRIEF", "ORB"} else list(keypoints)
    descriptor = create_descriptor(name)
    try:
        computed_keypoints, descriptors = descriptor.compute(gray, prepared_keypoints)
    except cv2.error:
        if descriptor_name == "BRIEF" and not BRIEF_AVAILABLE:
            computed_keypoints, descriptors = cv2.BRISK_create().detectAndCompute(gray, None)
        elif descriptor_name == "ORB":
            computed_keypoints, descriptors = cv2.ORB_create(nfeatures=1200).detectAndCompute(gray, None)
        else:
            raise
    return list(computed_keypoints or []), descriptors


def detect_and_compute_orb(gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    detector = cv2.ORB_create(nfeatures=1200)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return list(keypoints or []), descriptors


def ratio_test_matches(desc1: np.ndarray | None, desc2: np.ndarray | None, descriptor_name: str, ratio: float = 0.75) -> list[cv2.DMatch]:
    if desc1 is None or desc2 is None:
        return []
    matcher = cv2.BFMatcher(descriptor_norm(descriptor_name), crossCheck=False)
    matches = []
    for pair in matcher.knnMatch(desc1, desc2, k=2):
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            matches.append(first)
    return sorted(matches, key=lambda match: match.distance)


def cross_check_matches(desc1: np.ndarray | None, desc2: np.ndarray | None, descriptor_name: str) -> list[cv2.DMatch]:
    if desc1 is None or desc2 is None:
        return []
    matcher = cv2.BFMatcher(descriptor_norm(descriptor_name), crossCheck=True)
    return sorted(matcher.match(desc1, desc2), key=lambda match: match.distance)


def intersection_matches(matches_a: list[cv2.DMatch], matches_b: list[cv2.DMatch]) -> list[cv2.DMatch]:
    lookup = {(match.queryIdx, match.trainIdx): match for match in matches_b}
    return [match for match in matches_a if (match.queryIdx, match.trainIdx) in lookup]


def draw_matches(
    image1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    image2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    max_matches: int = 40,
    matches_mask: list[int] | None = None,
) -> np.ndarray:
    limited = matches[:max_matches]
    mask = matches_mask[: len(limited)] if matches_mask is not None else None
    return cv2.drawMatches(
        ensure_color(image1),
        keypoints1,
        ensure_color(image2),
        keypoints2,
        limited,
        None,
        matchesMask=mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def find_homography_from_matches(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(matches) < 4:
        return None, None
    src = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return cv2.findHomography(src, dst, cv2.RANSAC, 4.0)


def warp_corners(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, homography)


def bow_image_paths() -> list[tuple[str, Path]]:
    ensure_demo_data()
    paths: list[tuple[str, Path]] = []
    for class_dir in sorted(BOW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.glob("*.png")):
            paths.append((class_dir.name, image_path))
    return paths


def sift_descriptors(gray: np.ndarray) -> np.ndarray | None:
    _, descriptors = cv2.SIFT_create(nfeatures=500).detectAndCompute(gray, None)
    return descriptors


def build_bow_vocabulary(descriptor_sets: list[np.ndarray], cluster_count: int = 24) -> np.ndarray:
    stacked = np.vstack([desc for desc in descriptor_sets if desc is not None and len(desc) > 0]).astype(np.float32)
    cluster_count = min(cluster_count, len(stacked))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, _, centers = cv2.kmeans(stacked, cluster_count, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)


def encode_bow_histogram(descriptors: np.ndarray | None, vocabulary: np.ndarray) -> np.ndarray:
    histogram = np.zeros(len(vocabulary), dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return histogram
    distances = np.linalg.norm(descriptors[:, None, :] - vocabulary[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    for index in nearest:
        histogram[index] += 1.0
    histogram /= max(histogram.sum(), 1.0)
    return histogram
