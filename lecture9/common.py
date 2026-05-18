from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from skimage import data


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CIRCLESGRID_PATH = DATA_DIR / "circlesgrid.png"
LENA_PATH = DATA_DIR / "Lena.png"
TRAFFIC_PATH = DATA_DIR / "traffic.mp4"
PANORAMA_DIR = DATA_DIR / "panorama"
PANORAMA_MARKER_PATH = PANORAMA_DIR / ".dataset_source.txt"
PANORAMA_MAX_WIDTH = 1400
PANORAMA_SOURCE_URLS = [
    ("boat1.jpg", "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat1.jpg"),
    ("boat2.jpg", "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat2.jpg"),
    ("boat3.jpg", "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat3.jpg"),
    ("boat4.jpg", "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat4.jpg"),
    ("boat5.jpg", "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat5.jpg"),
    ("boat6.jpg", "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat6.jpg"),
]


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


def read_color(path: Path) -> np.ndarray:
    ensure_demo_data()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def download_image(url: str) -> np.ndarray:
    with urlopen(url, timeout=30) as response:
        payload = response.read()
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image from URL: {url}")
    return image


def resize_to_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = max_width / float(image.shape[1])
    return cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def generate_circlesgrid() -> None:
    if CIRCLESGRID_PATH.exists():
        return

    height = 320
    width = 320
    image = np.full((height, width, 3), 30, dtype=np.uint8)

    for row in range(7):
        for col in range(7):
            center = (40 + col * 40, 40 + row * 40)
            radius = 11 if (row + col) % 2 == 0 else 9
            cv2.circle(image, center, radius, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(image, center, radius + 3, (90, 160, 255), 2, cv2.LINE_AA)

    cv2.rectangle(image, (18, 18), (302, 302), (0, 210, 255), 2, cv2.LINE_AA)
    save_image(CIRCLESGRID_PATH, image)


def generate_lena() -> None:
    if LENA_PATH.exists():
        return

    for source in (
        ROOT_DIR.parent / "lecture3" / "data" / "lena.png",
        ROOT_DIR.parent / "lecture4" / "data" / "lena.png",
        ROOT_DIR.parent / "lecture3" / "data" / "Lena.png",
        ROOT_DIR.parent / "lecture4" / "data" / "Lena.png",
    ):
        if source.exists():
            shutil.copy2(source, LENA_PATH)
            print(f"Copied image: {source} -> {LENA_PATH}")
            return

    fallback = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    fallback = cv2.resize(fallback, (512, 512), interpolation=cv2.INTER_AREA)
    save_image(LENA_PATH, fallback)


def generate_traffic_clip(frame_count: int = 150) -> None:
    if TRAFFIC_PATH.exists():
        return

    source = ROOT_DIR.parent / "lecture8" / "data" / "traffic.mp4"
    if source.exists():
        capture = cv2.VideoCapture(str(source))
        if capture.isOpened():
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                fps = fps if fps > 0 else 15.0
                source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame = min(max(0, source_frames // 10), max(0, source_frames - frame_count - 1))
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                ok, first = capture.read()
                if ok and first is not None:
                    resized_first = cv2.resize(first, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    height, width = resized_first.shape[:2]
                    writer = cv2.VideoWriter(
                        str(TRAFFIC_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        min(fps, 15.0),
                        (width, height),
                    )
                    if writer.isOpened():
                        written = 0
                        frame = first
                        while written < frame_count and frame is not None:
                            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                            writer.write(resized)
                            written += 1
                            ok, frame = capture.read()
                            if not ok:
                                break
                        writer.release()
                        if written > 0:
                            print(f"Created video: {TRAFFIC_PATH}")
                            return
            finally:
                capture.release()

    base = cv2.cvtColor(data.rocket(), cv2.COLOR_RGB2BGR)
    base = cv2.resize(base, (640, 360), interpolation=cv2.INTER_AREA)
    writer = cv2.VideoWriter(str(TRAFFIC_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    if not writer.isOpened():
        raise OSError(f"Could not create video: {TRAFFIC_PATH}")
    try:
        for index in range(frame_count):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            shift = index * 3
            x0 = 40 + shift % 560
            frame[:, :] = (35, 35, 35)
            cv2.line(frame, (320, 0), (320, 359), (70, 70, 70), 8, cv2.LINE_AA)
            cv2.rectangle(frame, (x0, 150), (x0 + 110, 250), (80, 120, 220), -1, cv2.LINE_AA)
            cv2.circle(frame, (x0 + 25, 255), 16, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, (x0 + 85, 255), 16, (20, 20, 20), -1, cv2.LINE_AA)
            patch = base[90:240, 200:440]
            frame[100:250, 200:440] = cv2.addWeighted(frame[100:250, 200:440], 0.35, patch, 0.65, 0.0)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created video: {TRAFFIC_PATH}")


def generate_panorama_images() -> None:
    target_paths = [PANORAMA_DIR / name for name, _ in PANORAMA_SOURCE_URLS]
    if PANORAMA_MARKER_PATH.exists() and all(path.exists() for path in target_paths):
        return

    PANORAMA_DIR.mkdir(parents=True, exist_ok=True)
    for legacy_path in (PANORAMA_DIR / "0.jpg", PANORAMA_DIR / "1.jpg", PANORAMA_DIR / "2.jpg"):
        if legacy_path.exists():
            legacy_path.unlink()

    for filename, url in PANORAMA_SOURCE_URLS:
        image = download_image(url)
        image = resize_to_max_width(image, PANORAMA_MAX_WIDTH)
        save_image(PANORAMA_DIR / filename, image)

    PANORAMA_MARKER_PATH.write_text(
        "OpenCV stitching sample dataset: boat1-boat6\n"
        "Source: https://github.com/opencv/opencv_extra/tree/4.x/testdata/stitching\n",
        encoding="utf-8",
    )


def panorama_image_paths() -> list[Path]:
    ensure_demo_data()
    return [PANORAMA_DIR / name for name, _ in PANORAMA_SOURCE_URLS]


def create_stitcher():
    if hasattr(cv2, "Stitcher_create"):
        return cv2.Stitcher_create()
    return cv2.createStitcher(False)


def ensure_demo_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_circlesgrid()
    generate_lena()
    generate_traffic_clip()
    generate_panorama_images()
