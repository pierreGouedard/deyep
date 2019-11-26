# Global import
import numpy as np
import cv2
from matplotlib import pyplot as plt
import itertools


# Local import


def build_filter_bank(l_ksizes):
    l_filters = []

    for ksize in l_ksizes:
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, 0., 2.0, 2., 0, ktype=cv2.CV_32F)

        kern /= kern.sum()

        l_filters.append(kern)

    return l_filters


class GaborFilter(object):

    def __init__(self, l_sigmas, l_thetas, l_lambdas):

        # Get list of
        self.sigmas = l_sigmas
        self.thetas = l_thetas
        self.lambdas = l_lambdas

        # Encode dictionary of Gabor filters
        self.filters = self.encode_filters(self.sigmas, self.lambdas, self.thetas)

    @staticmethod
    def encode_filters(l_sigmas, l_lambdas, l_thetas):

        d_filters = {}
        for sgm, lmbd, tht in itertools.product(l_sigmas, l_lambdas, l_thetas):
            d_filters['']


class SquareDrawer(object):

    def __init__(self, size, n, img_dim, coords=None, scale=1., seed=None):

        # dim of the square
        self.size = size
        self.n = n
        self.scale = scale

        # Init image and shape
        self.image = np.zeros(img_dim)
        self.scaled_image = np.zeros(img_dim)
        self.patch = np.ones((size, size))

        # Init seed
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 10000)

        np.random.seed(self.seed)

        # Init positions
        if coords is not None:
            self.coords = [(x, y) for (x, y) in coords]

        else:
            l_x = np.random.randint(0 + (self.size / self.scale), self.image.shape[0] - (self.size / self.scale), n)
            l_y = np.random.randint(0 + (self.size / self.scale), self.image.shape[1] - (self.size / self.scale), n)
            self.coords = [(x, y) for x, y in zip(*[l_x, l_y])]

        # Draw original image
        self.draw(self.image, self.coords, self.size)

    @staticmethod
    def draw(img, coords, size, scale=1., translate_x=0, translate_y=0):

        # Build patch
        p = np.ones((int(size * scale), int(size * scale)))

        # Draw patch at given positions
        for x, y in coords:
            img[
                int(x * scale) - translate_x: int(x * scale) + p.shape[0] - translate_x,
                int(y * scale) - translate_y: int(y * scale) + p.shape[0] - translate_y
            ] = p

    def scale_up(self, scale):

        # Create new patch
        p = np.ones((int(self.size * scale), int(self.size * scale)))

        # Get ideal translation to recenter up scale square
        upper_left, translate_x, translate_y = list(self.image.shape), 0, 0
        for (x, y) in self.coords:
            if x < upper_left[0]:
                upper_left[0] = x
                translate_x = int(x * scale - x)

            if y < upper_left[1]:
                upper_left[1] = y
                translate_y = int(y * scale - y)

        # Draw scaled image
        self.draw(
            self.scaled_image, self.coords, self.size, scale=scale, translate_x=translate_x, translate_y=translate_y
        )

        return self

    def show_images(self):
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(self.image, cmap='gray')

        plt.subplot(122)
        plt.imshow(self.scaled_image, cmap='gray')
        plt.show()

        return self


class MosaicDrawer(object):

    def __init__(self, img_dim, treshold, wavelength, orientation, scale=1.):

        self.scale = scale
        self.theta = orientation
        self.treshold = treshold
        self.wavelength = wavelength

        # Init image and shape
        self.image = np.zeros(img_dim)
        self.scaled_image = np.zeros(img_dim)

        # Draw original image
        self.draw(self.image, self.wavelength, self.theta, self.treshold)

    @staticmethod
    def draw(img, wavelength, theta, treshold, scale=1.):

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                v = np.cos((2. * np.pi / (wavelength * scale)) * ((x * np.cos(theta)) + (y * np.sin(theta))))

                if v > treshold:
                    img[x, y] = 1.

                v = np.cos((2. * np.pi / (wavelength * scale)) * ((x * np.sin(theta)) + (y * np.cos(theta))))
                if v > treshold:
                    img[x, y] = 1.

    def scale_up(self, scale):
        self.draw(self.scaled_image, self.wavelength, self.theta, self.treshold, scale=scale)
        return self

    def show_images(self):
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(self.image, cmap='gray')

        plt.subplot(122)
        plt.imshow(self.scaled_image, cmap='gray')
        plt.show()

        return self


if __name__ == '__main__':

    # 1. plot diferent size kernel
    # l_filters = build_filter_bank([100, 200, 300, 400, 500])
    #
    # for filter in l_filter:
    #     plt.imshow(filter, cmap='gray')
    #     plt.show()

    # 2.A simple randomly positioned square
    sd = SquareDrawer(5, 10, (500, 500), scale=0.05, seed=123)\
        .scale_up(1.55)\
        .show_images()

    # 2.B draw mosaic
    sd = MosaicDrawer((500, 500), 0.9, 20., 0.3, scale=0.05)\
        .scale_up(1.55)\
        .show_images()
