# Global import
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Local import


def build_filter_bank(l_ksizes):
    l_filters = []

    for ksize in l_ksizes:
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, 0., 2.0, 2., 0, ktype=cv2.CV_32F)

        kern /= kern.sum()

        l_filters.append(kern)

    return l_filters


class SquareDrawer():

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

    def draw_random_squares(self):

        for x, y in self.coords:
            self.image[x: x + self.size, y: y + self.size] = self.patch

        return self

    def scale_up(self, scale):
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

        for x, y in self.coords:

            self.scaled_image[
            int(x * scale) - translate_x: int(x * scale) + p.shape[0] - translate_x,
            int(y * scale) - translate_y: int(y * scale) + p.shape[0] - translate_y
            ] = p

        return self

    def swipe_images(self):
        tmp_image = self.scaled_image.copy()
        self.scaled_image = self.image.copy()
        self.image = tmp_image

        return self

    def show_images(self):
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(self.image, cmap='gray')

        plt.subplot(212)
        plt.imshow(self.scaled_image, cmap='gray')
        plt.show()

    def show_image(self, key):

        if key == 'orig':
            plt.imshow(self.scaled_image, cmap='gray')

        elif key == 'scaled':
            plt.imshow(self.scaled_image, cmap='gray')

        plt.show()

        return self


class MosaicDrawer():

    def __init__(self, img_dim, treshold, wavelength, orientation, scale=1.):

        self.scale = scale
        self.theta = orientation
        self.treshold = treshold
        self.wavelength = wavelength

        # Init image and shape
        self.image = np.zeros(img_dim)
        self.scaled_image = np.zeros(img_dim)

    def draw_mosaic(self):

        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                v = np.cos((2. * np.pi / self.wavelength) * ((x * np.cos(self.theta)) + (y * np.sin(self.theta))))

                if v > self.treshold:
                    self.image[x, y] = 1.

                v = np.cos((2. * np.pi / self.wavelength) * ((x * np.sin(self.theta)) + (y * np.cos(self.theta))))
                if v > self.treshold:
                    self.image[x, y] = 1.

        return self

    def show_images(self):
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(self.image, cmap='gray')

        plt.subplot(212)
        plt.imshow(self.scaled_image, cmap='gray')
        plt.show()

    def show_image(self, key):

        if key == 'orig':
            plt.imshow(self.image, cmap='gray')

        elif key == 'scaled':
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

    # # 2. Generate different scale simple pattern on image
    # # 2.A simple randomly positioned square
    # sd = SquareDrawer(5, 10, (500, 500), scale=0.05, seed=123)\
    #     .draw_random_squares()
    #
    # sd.scale_up(1.55)\
    #     .show_images()

    # 2.B draw mosaic
    sd = MosaicDrawer((500, 500), 0.7, 20., 0.3, scale=0.05)\
        .draw_mosaic()\
        .show_image(key='orig')


    import IPython
    IPython.embed()

