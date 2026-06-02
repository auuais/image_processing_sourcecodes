from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from common import digits_image_path, draw_text_block, finalize_figure, load_grayscale_image, plot_bgr, plot_gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lecture 12 page 68: OCR using KNN and SVM.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving.")
    return parser.parse_args()


def split_digits_sheet(sheet: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells = [np.hsplit(row, 100) for row in np.vsplit(sheet, 50)]
    cell_array = np.array(cells)
    train_digits = cell_array[:, :50].reshape(-1, 20, 20)
    test_digits = cell_array[:, 50:100].reshape(-1, 20, 20)
    train_labels = np.repeat(np.arange(10), 250).astype(np.int32)
    test_labels = np.repeat(np.arange(10), 250).astype(np.int32)
    return train_digits, test_digits, train_labels, test_labels


def build_hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(20, 20),
        _blockSize=(8, 8),
        _blockStride=(4, 4),
        _cellSize=(4, 4),
        _nbins=9,
    )


def compute_hog_features(digits: np.ndarray, hog: cv2.HOGDescriptor) -> np.ndarray:
    return np.array([hog.compute(digit).reshape(-1) for digit in digits], dtype=np.float32)


def build_confusion_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int32)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def render_confusion(ax, matrix: np.ndarray, title: str) -> None:
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    threshold = matrix.max() * 0.55 if matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] >= threshold else "black"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7, color=color)


def build_prediction_grid(
    digits: np.ndarray,
    truth: np.ndarray,
    knn_prediction: np.ndarray,
    svm_prediction: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    tile_width = 100
    tile_height = 120
    columns = 4
    rows = int(np.ceil(len(indices) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)

    for slot, sample_index in enumerate(indices):
        row = slot // columns
        column = slot % columns
        x0 = column * tile_width
        y0 = row * tile_height

        digit = cv2.resize(digits[sample_index], (56, 56), interpolation=cv2.INTER_NEAREST)
        digit_bgr = cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)
        canvas[y0 + 8 : y0 + 64, x0 + 22 : x0 + 78] = digit_bgr

        gt = int(truth[sample_index])
        knn_value = int(knn_prediction[sample_index])
        svm_value = int(svm_prediction[sample_index])

        cv2.putText(canvas, f"T:{gt}", (x0 + 12, y0 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"K:{knn_value}",
            (x0 + 12, y0 + 101),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 140, 0) if knn_value == gt else (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"S:{svm_value}",
            (x0 + 12, y0 + 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 140, 0) if svm_value == gt else (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
    return canvas


def top_confusions(matrix: np.ndarray, limit: int = 5) -> list[str]:
    mistakes: list[tuple[int, int, int]] = []
    for true_label in range(matrix.shape[0]):
        for predicted_label in range(matrix.shape[1]):
            if true_label == predicted_label or matrix[true_label, predicted_label] == 0:
                continue
            mistakes.append((int(matrix[true_label, predicted_label]), true_label, predicted_label))
    mistakes.sort(reverse=True)
    return [f"{true_label}->{predicted_label}: {count}" for count, true_label, predicted_label in mistakes[:limit]]


def main() -> None:
    args = parse_args()
    sheet = load_grayscale_image(digits_image_path())
    train_digits, test_digits, train_labels, test_labels = split_digits_sheet(sheet)

    hog = build_hog_descriptor()
    train_features = compute_hog_features(train_digits, hog)
    test_features = compute_hog_features(test_digits, hog)

    knn = cv2.ml.KNearest_create()
    knn.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    _, knn_prediction, _, _ = knn.findNearest(test_features, 5)
    knn_prediction = knn_prediction.reshape(-1).astype(np.int32)

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(12.5)
    svm.setGamma(0.50625)
    svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    svm_prediction = svm.predict(test_features)[1].reshape(-1).astype(np.int32)

    knn_accuracy = float(np.mean(knn_prediction == test_labels))
    svm_accuracy = float(np.mean(svm_prediction == test_labels))
    knn_confusion = build_confusion_matrix(test_labels, knn_prediction)
    svm_confusion = build_confusion_matrix(test_labels, svm_prediction)

    error_indices = np.where((knn_prediction != test_labels) | (svm_prediction != test_labels))[0].tolist()
    sample_indices = [digit * 250 + 12 for digit in range(10)]
    sample_indices.extend(error_indices[:2] if error_indices else [120, 1120])
    prediction_grid = build_prediction_grid(test_digits, test_labels, knn_prediction, svm_prediction, sample_indices[:12])

    fig = plt.figure(figsize=(18, 10))
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.95])

    plot_gray(fig.add_subplot(grid[0, 0]), sheet, "OpenCV digits sheet")
    plot_bgr(fig.add_subplot(grid[1, 0]), prediction_grid, "Sample predictions (T=true, K=KNN, S=SVM)")
    render_confusion(fig.add_subplot(grid[0, 1]), knn_confusion, f"KNN confusion ({knn_accuracy:.2%})")
    render_confusion(fig.add_subplot(grid[1, 1]), svm_confusion, f"SVM confusion ({svm_accuracy:.2%})")

    summary = (
        f"Training samples: {len(train_labels)}\n"
        f"Test samples: {len(test_labels)}\n"
        f"HOG feature length: {train_features.shape[1]}\n\n"
        f"KNN accuracy: {knn_accuracy:.2%}\n"
        f"SVM accuracy: {svm_accuracy:.2%}\n\n"
        f"Top KNN confusions:\n"
        f"{chr(10).join(top_confusions(knn_confusion) or ['none'])}\n\n"
        f"Top SVM confusions:\n"
        f"{chr(10).join(top_confusions(svm_confusion) or ['none'])}"
    )
    draw_text_block(fig.add_subplot(grid[:, 2]), "OCR Summary", summary)
    finalize_figure(fig, "page68_optical_character_recognition_using_knn_and_svm.png", show=args.show)


if __name__ == "__main__":
    main()
