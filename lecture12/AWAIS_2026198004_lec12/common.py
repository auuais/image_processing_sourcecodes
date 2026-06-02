from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

PEDESTRIAN_DIR = DATA_DIR / "pedestrians"
OCR_DIR = DATA_DIR / "ocr"
FACE_DIR = DATA_DIR / "faces"
ARUCO_DIR = DATA_DIR / "aruco"
NEW_DATA_DIR = DATA_DIR / "new_data"

PEDESTRIAN_URLS = {
    "basketball1.png": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/basketball1.png",
    "basketball2.png": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/basketball2.png",
}
OCR_DIGITS_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/digits.png"
LENA_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
LBP_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/lbpcascades/lbpcascade_frontalface_improved.xml"

FACE_SCENE_PATH = FACE_DIR / "synthetic_group.png"
LENA_PATH = FACE_DIR / "lena.jpg"
LBP_CASCADE_PATH = FACE_DIR / "lbpcascade_frontalface_improved.xml"
OCR_DIGITS_PATH = OCR_DIR / "digits.png"
ARUCO_SCENE_PATH = ARUCO_DIR / "synthetic_markers.png"

ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_MARKER_IDS = [5, 12, 23, 31]


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


def draw_text_block(ax, title: str, text: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)


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


def download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read()


def download_image(url: str) -> np.ndarray:
    buffer = np.frombuffer(download_bytes(url), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image from URL: {url}")
    return image


def write_downloaded_file(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(download_bytes(url))
    print(f"Saved file: {destination}")
    return destination


def write_downloaded_image(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(destination, download_image(url))
    return destination


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def ensure_base_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_marker_image(dictionary, marker_id: int, side_pixels: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels)
    marker = np.zeros((side_pixels, side_pixels), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, side_pixels, marker, 1)
    return marker


def build_face_scene() -> np.ndarray:
    lena = load_color_image(LENA_PATH)
    canvas = np.full((1400, 1400, 3), 236, dtype=np.uint8)
    cv2.rectangle(canvas, (30, 30), (1370, 1370), (248, 248, 248), -1)
    cv2.rectangle(canvas, (30, 30), (1370, 1370), (215, 215, 215), 4, cv2.LINE_AA)

    placements = [
        (50, 80, 0.70),
        (720, 90, 0.80),
        (300, 720, 0.90),
    ]
    for x, y, scale in placements:
        face = cv2.resize(lena, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        height, width = face.shape[:2]
        shadow = canvas[y + 18 : y + 18 + height, x + 18 : x + 18 + width]
        shadow[:] = (shadow.astype(np.float32) * 0.85).astype(np.uint8)
        canvas[y : y + height, x : x + width] = face
    return canvas


def build_aruco_scene() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY_NAME))
    canvas = np.full((900, 1200, 3), 248, dtype=np.uint8)

    for row in range(canvas.shape[0]):
        shade = 236 + int(12 * row / max(canvas.shape[0] - 1, 1))
        canvas[row, :, :] = shade
    for x in range(80, canvas.shape[1], 160):
        cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), (235, 235, 235), 1, cv2.LINE_AA)
    for y in range(80, canvas.shape[0], 160):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (235, 235, 235), 1, cv2.LINE_AA)

    placements = [
        (60, 80, 5, 180, -12.0),
        (760, 80, 12, 160, 8.0),
        (120, 520, 23, 190, -6.0),
        (760, 500, 31, 170, 14.0),
    ]
    for x, y, marker_id, size, angle in placements:
        marker = generate_marker_image(dictionary, marker_id, size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        rotation = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(marker_bgr, rotation, (size, size), borderValue=(255, 255, 255))
        canvas[y : y + size, x : x + size] = rotated
        cv2.putText(
            canvas,
            f"ID {marker_id}",
            (x, y + size + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
    return canvas


def ensure_pedestrian_dataset() -> None:
    ensure_base_dirs()
    for name, url in PEDESTRIAN_URLS.items():
        write_downloaded_image(url, PEDESTRIAN_DIR / name)


def ensure_ocr_dataset() -> None:
    ensure_base_dirs()
    write_downloaded_image(OCR_DIGITS_URL, OCR_DIGITS_PATH)


def ensure_face_dataset() -> None:
    ensure_base_dirs()
    write_downloaded_image(LENA_URL, LENA_PATH)
    write_downloaded_file(LBP_CASCADE_URL, LBP_CASCADE_PATH)
    if not FACE_SCENE_PATH.exists():
        save_image(FACE_SCENE_PATH, build_face_scene())


def ensure_aruco_dataset() -> None:
    ensure_base_dirs()
    if not ARUCO_SCENE_PATH.exists():
        save_image(ARUCO_SCENE_PATH, build_aruco_scene())


def ensure_demo_data() -> None:
    ensure_base_dirs()
    ensure_pedestrian_dataset()
    ensure_ocr_dataset()
    ensure_face_dataset()
    ensure_aruco_dataset()


def pedestrian_image_paths() -> list[Path]:
    ensure_pedestrian_dataset()
    preferred_paths = []
    for candidate in ["people.jpg", "people.jpeg", "people.png"]:
        path = NEW_DATA_DIR / candidate
        if path.exists():
            preferred_paths.append(path)
    default_paths = [PEDESTRIAN_DIR / name for name in sorted(PEDESTRIAN_URLS)]
    return preferred_paths + default_paths


def digits_image_path() -> Path:
    ensure_ocr_dataset()
    return OCR_DIGITS_PATH


def face_scene_path() -> Path:
    ensure_face_dataset()
    return FACE_SCENE_PATH


def haar_cascade_path() -> Path:
    user_path = NEW_DATA_DIR / "haarcascade_frontalface_default.xml"
    if user_path.exists():
        return user_path
    return Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def lbp_cascade_path() -> Path:
    user_path = NEW_DATA_DIR / "lbpcascade_frontalface.xml"
    if user_path.exists():
        return user_path
    ensure_face_dataset()
    return LBP_CASCADE_PATH


def aruco_scene_path() -> Path:
    ensure_aruco_dataset()
    return ARUCO_SCENE_PATH


def face_video_path() -> Path | None:
    candidate = NEW_DATA_DIR / "faces.mp4"
    if candidate.exists():
        return candidate
    return None
