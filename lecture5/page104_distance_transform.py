from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, normalize_channel, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = np.full((480, 640), 255, np.uint8)
    cv2.circle(image, (320, 240), 100, 0, -1)

    distmap = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_gray(axes[0], image, "Binary image")
    plot_gray(axes[1], normalize_channel(distmap), "Distance transform")

    finalize_figure(fig, "page104_distance_transform.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 104 - Distance transform"))
