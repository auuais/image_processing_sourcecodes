from __future__ import annotations

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (
    bow_image_paths,
    build_bow_vocabulary,
    encode_bow_histogram,
    finalize_figure,
    load_color_image,
    parse_show_flag,
    plot_bgr,
    sift_descriptors,
)


def nearest_centroid_accuracy(histograms, labels):
    grouped = defaultdict(list)
    for hist, label in zip(histograms, labels):
        grouped[label].append(hist)
    centroids = {label: np.mean(items, axis=0) for label, items in grouped.items()}
    correct = 0
    for hist, label in zip(histograms, labels):
        predicted = min(centroids, key=lambda name: np.linalg.norm(hist - centroids[name]))
        correct += int(predicted == label)
    return correct / max(len(labels), 1)


def main(show: bool = False) -> None:
    image_records = bow_image_paths()
    descriptor_sets = []
    histograms = []
    labels = []
    sample_paths = {}

    for label, image_path in image_records:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        descriptors = sift_descriptors(gray)
        descriptor_sets.append(descriptors)
        labels.append(label)
        sample_paths.setdefault(label, image_path)

    vocabulary = build_bow_vocabulary(descriptor_sets, cluster_count=24)
    for descriptors in descriptor_sets:
        histograms.append(encode_bow_histogram(descriptors, vocabulary))

    class_hist = defaultdict(list)
    for label, hist in zip(labels, histograms):
        class_hist[label].append(hist)
    class_means = {label: np.mean(items, axis=0) for label, items in class_hist.items()}
    accuracy = nearest_centroid_accuracy(histograms, labels)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ordered_labels = sorted(class_means)
    for index, label in enumerate(ordered_labels):
        sample_image = cv2.imread(str(sample_paths[label]), cv2.IMREAD_COLOR)
        plot_bgr(axes[0, index], sample_image, f"{label} sample")
        axes[1, index].bar(np.arange(len(class_means[label])), class_means[label], color="#2a9d8f")
        axes[1, index].set_title(f"{label} mean BoW histogram")
        axes[1, index].set_xlabel("Visual word")
        axes[1, index].set_ylabel("Normalized count")

    fig.suptitle(f"BoW model for global image descriptor - nearest-centroid accuracy: {accuracy:.1%}", fontsize=14)
    finalize_figure(fig, "page90_bow_global_descriptor.png", show=show)


if __name__ == "__main__":
    main(show=parse_show_flag("Lecture 8 page 90 - BoW model for global image descriptor"))
