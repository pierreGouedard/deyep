# Global import
import cv2 as cv


def apply_threshold(ax_img, thresh):
    # Convert to gray scale
    ax_grey_img = cv.cvtColor(ax_img, cv.COLOR_BGR2GRAY)
    ret, ax_thresh_img = cv.threshold(ax_grey_img, thresh, 1, cv.THRESH_BINARY_INV)
    return ax_thresh_img.astype(bool)