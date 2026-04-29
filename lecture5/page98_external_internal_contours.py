from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import finalize_figure, load_gray_image, parse_show_flag, plot_gray


def main(show: bool = False) -> None:
    image = load_gray_image("BnW.png")
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    external = np.zeros_like(image)
    internal = np.zeros_like(image)

    for index in range(len(contours)):
        if hierarchy[0][index][3] == -1:
            cv2.drawContours(external, contours, index, 255, -1)
        else:
            cv2.drawContours(internal, contours, index, 255, -1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_gray(axes[0], image, "Original")
    plot_gray(axes[1], external, "External contours")
    plot_gray(axes[2], internal, "Internal contours")

    finalize_figure(fig, "page98_external_internal_contours.png", show)


if __name__ == "__main__":
    main(parse_show_flag("Page 98 - External and internal contours"))
