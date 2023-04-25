# Global import
import numpy as np
from scipy.sparse import lil_matrix
import cv2 as cv
from matplotlib import pyplot as plt

# Local import
from deyep.linalg.spmat_op import fill_gap
from deyep.models.image import SimpleChain

# TODO: no need for a fucking class there
class ShapeDrawer:
    def __init__(self, im_shape, factor=0.5):
        self.im_shape = im_shape
        self.factor = 3

    def draw_shape(self, chain: SimpleChain) -> None:
        ax_shape = np.zeros(self.im_shape)
        ax_shape += cv.drawContours(np.zeros(self.im_shape, dtype=np.uint8), [chain.points()[:, ::-1]], 0, 255, -1)
        plt.imshow(ax_shape)
        plt.show()
