# Global import
import sys
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from PIL import Image
import cv2 as cv
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, hstack

import sys

yala_path = Path(__file__).parent.parent.parent
sys.path.append(yala_path.as_posix())


# Deyep
from deyep.core.firing_graph import DeyepFiringGraph
from deyep.core.data_models import FgComponents, BitMap
from deyep.core.shape_encoder import ShapeEncoder
from deyep.core.shape_drawer import ShapeDrawer
from deyep.core.img_encoder import ImgCoordEncoder

class ShapeEncoding:

    def __init__(self, n_features: int, mode: str = 'normal'):
        self.n_features = n_features
        self.mode = mode

    def encode_image(self, ax_image: np.ndarray):
        # TODO:
        #   - build CH of 1 f(target, features)
        #   - encode shape =>
        #       * Naive just the CH =>
        #           * Try redraw shape from encoding
        #       * dichotomous algorithm

        # Transform image
        img_encoder = ImgCoordEncoder(max(ax_image.shape), self.n_features, 1, 0.0)
        X, y = img_encoder.fit_transform(ax_image)
        bitmap = BitMap(img_encoder.bf_map)

        # get CH shape as firing graph
        sax_i = X.T.dot(y[:, 1])
        comp = FgComponents(inputs=sax_i, mask_inputs=sax_i, partitions=[{'id': 1}], levels=np.array([self.n_features]))

        # Plot CH
        fg = DeyepFiringGraph.from_comp(comp)
        self.show_image(fg.seq_propagate(X).A.reshape(ax_image.shape))

        shape_encoder = ShapeEncoder(bitmap)
        ax_code = shape_encoder.encode_shape(X, comp)
        self.plot_shape_encoding(ax_code[0])
        plt.show()

        # Draw shape from encoding
        shape_drawer = ShapeDrawer(bitmap, ax_image.shape)
        sax_i = shape_drawer.draw_shape(img_encoder, np.vstack([ax_code, ax_code]))

        comp = FgComponents(inputs=sax_i, mask_inputs=sax_i, partitions=[{'id': 1}, {"id": 2}], levels=np.array([self.n_features] * 2))
        fg = DeyepFiringGraph.from_comp(comp)
        self.show_image(fg.seq_propagate(X).A[:, [0]].reshape(ax_image.shape))
        import IPython
        IPython.embed()
    def show_image(self, ax_image: np.ndarray, title: str = "signal image"):
        plt.figure(figsize=(20, 10))
        plt.imshow(ax_image)
        plt.title(title)
        plt.show()

    def plot_shape_encoding(self, ax_code):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.bar(list(map(str, range(len(ax_code)))), ax_code)
        plt.show()




def apply_threshold(ax_img, thresh):

    # Convert to gray scale
    ax_grey_img = cv.cvtColor(ax_img, cv.COLOR_BGR2GRAY)
    ret, ax_thresh_img = cv.threshold(ax_grey_img, thresh, 1, cv.THRESH_BINARY_INV)
    return ax_thresh_img.astype(bool)


if __name__ == '__main__':
    """
    Usage:
    python deyep/scripts/encoding.py

    """
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    numbers_path = data_path / "numbers"

    l_filenames = ['2_normal.png', '2_flip.png', '2_rotated.png', '2_translated.png']

    # Parameters
    threshold = 100
    n_features = 20

    for filename in l_filenames:
        # Read image
        image = Image.open(numbers_path / filename)

        # Apply treshold to isolate shape
        masked_image = apply_threshold(np.asarray(image), threshold)

        # Now start encoding
        shape_encoder = ShapeEncoding(n_features)
        shape_encoder.encode_image(masked_image)



