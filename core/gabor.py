# Global import
import numpy as np
import cv2
from matplotlib import pyplot as plt
import itertools

# Local import
from core.names import KVName


def build_filter_bank(l_ksizes):

    l_filters = []

    for ksize in l_ksizes:
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, 0., 2.0, 2., 0, ktype=cv2.CV_32F)

        kern /= kern.sum()

        l_filters.append(kern)

    return l_filters


class GaborFilter(object):

    def __init__(self, ksize, n_sigma, l_thetas, l_lambdas, l_parts, l_channels=None):

        # Get minimum kernel size
        self.ksize = ksize

        # Get list of parameters
        self.lambdas = l_lambdas
        self.sigmas = self.generate_sigmas(l_lambdas, n_sigma)
        self.thetas = l_thetas
        self.channels = l_channels
        self.parts = l_parts

        # Encode dictionary of Gabor filters
        self.filters, self.keys = self.encode_filters(
            self.sigmas, self.lambdas, self.thetas, self.parts, l_channels=self.channels
        )

    @staticmethod
    def generate_sigmas(l_lambdas, n_sigma, rnd=2):

        l_sigmas = []
        for lmbd in l_lambdas:
            l_sigmas.append([])
            start, stop = float(lmbd) / (2 * np.pi), float(lmbd) / (2 * np.pi) * 10
            for sigma in np.arange(start, stop, (stop - start) / n_sigma):
                l_sigmas[-1].append(float(int(sigma * pow(10, rnd))) / pow(10, rnd))

        return l_sigmas

    @staticmethod
    def encode_filters(l_sigmas, l_lambdas, l_thetas, l_parts, l_channels=None):

        # Get every combination of different parameters
        t_names = ['sgm', 'lmbd', 'tht', 'part']
        if l_channels is not None:
            it_params = itertools.product(l_thetas, l_parts, l_channels)
            t_names.append('chnnls')
        else:
            it_params = itertools.product(l_thetas, l_parts)

        # Encode filters
        d_filters, l_keys = {}, []
        for t_params in it_params:
            for i, lmbd in enumerate(l_lambdas):
                for sigma in l_sigmas[i]:

                    # Get params
                    d_params = {k: v for k, v in zip(t_names, [sigma, lmbd] + list(t_params))}

                    # Update filters
                    l_keys.append(KVName.from_dict(d_params).to_string())
                    d_filters[l_keys[-1]] = d_params.copy()

        return d_filters, sorted(l_keys)

    def generate_filters_multi_scale(self, l_scales, key):
        l_filters = []

        for scale in l_scales:
            l_filters.append(self.generate_filters(scale, [key]))

        return l_filters

    def generate_filters(self, scale, l_keys=None):

        for key in self.keys:

            if l_keys is not None:
                if key not in l_keys:
                    continue

            yield key, self.generate_filter(self.ksize, scale, **self.filters[key])

    @staticmethod
    def generate_filter(ksize, scale, **params):

        # Build kernel
        kern = cv2.getGaborKernel(
            (int(ksize * scale), int(ksize * scale)),
            sigma=params['sgm'] * scale,
            theta=params['tht'],
            lambd=params['lmbd'] * scale,
            gamma=0.25,
            psi=0 if params.get('part', 're') == 're' else np.pi * 0.5,
            ktype=cv2.CV_32F
        )

        return kern


class SquareDrawer(object):

    def __init__(self, size, n, img_dim, coords=None, scale=1., seed=None, shift=False):

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
            l_x = np.random.randint(0 + (img_dim[0] / 4), self.image.shape[0] - (img_dim[0] / 4), n)
            l_y = np.random.randint(0 + (img_dim[0] / 4), self.image.shape[1] - (img_dim[0] / 4), n)
            self.coords = [(x, y) for x, y in zip(*[l_x, l_y])]

            if shift:
                x_min, y_min = min(l_x), min(l_y)
                self.coords = [(x - x_min, y - y_min) for x, y in self.coords]

        # Draw original image
        self.draw(self.image, self.coords, self.size)

    @staticmethod
    def draw(img, coords, size, scale=1.):

        # Build patch
        p = np.ones((int(size * scale), int(size * scale)))

        # Draw patch at given positions
        for x, y in coords:
            x_start, x_end, y_start, y_end = SquareDrawer.get_patch_index(
                x, y, p.shape[1], p.shape[0], scale, img.shape
            )
            img[y_start: y_end, x_start: x_end] = p

    @staticmethod
    def get_patch_index(x, y, w, h, scale, max_dims):
        return int(x * scale), min(int(x * scale) + w, max_dims[1]), int(y * scale), min(int(y * scale) + h, max_dims[0])

    def scale_up(self, scale):

        # Get ideal translation to recenter up scale square
        (h, w), l_coords = self.image.shape, []

        for (x, y) in self.coords:
            if x * scale < w and y * scale < h:
                l_coords.append((x, y))

        # Draw scaled image
        self.draw(self.scaled_image, l_coords, self.size, scale=scale)

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

    # 1. plot different size kernel
    gabor_bank = GaborFilter(
        ksize=49, n_sigma=5, l_thetas=range(0, 181, 30), l_lambdas=[5, 10, 15, 20, 30, 40], l_parts=['re', 'im']
    )
    # Get a filter at different scale
    l_filters = list(gabor_bank.generate_filters(scale=1, l_keys=[gabor_bank.keys[111]]))
    l_filters += list(gabor_bank.generate_filters(scale=2, l_keys=[gabor_bank.keys[111]]))

    # Plot filters
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(l_filters[0][1], cmap='gray')
    plt.subplot(122)
    plt.imshow(l_filters[1][1], cmap='gray')
    plt.show()

    # 2.A simple randomly positioned square
    sd = SquareDrawer(5, 10, (500, 500), scale=0.05, seed=123)\
        .scale_up(1.55)\
        .show_images()

    # 2.B draw mosaic
    md = MosaicDrawer((500, 500), 0.9, 20., 0.3, scale=0.05)\
        .scale_up(1.55)\
        .show_images()


