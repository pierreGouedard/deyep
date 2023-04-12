# Global import
from typing import List
import numpy as np
import cv2 as cv


def apply_threshold(ax_img: np.ndarray, thresh: float) -> np.ndarray:
    # Convert to gray scale
    ax_grey_img = cv.cvtColor(ax_img, cv.COLOR_BGR2GRAY)
    ret, ax_thresh_img = cv.threshold(ax_grey_img, thresh, 1, cv.THRESH_BINARY_INV)
    return ax_thresh_img.astype(bool)


def get_mask_cc(ax_bin: np.ndarray, comp_min_size: int) -> List[np.ndarray]:
    n_cc, ax_labels, stats, _ = cv.connectedComponentsWithStats(ax_bin, connectivity=8)
    return [ax_labels == (i + 1) for i, ok in enumerate(stats[1:, -1] > comp_min_size) if ok]

