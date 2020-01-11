# Global imports
import unittest
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import correlate
from scipy.stats import entropy
import random

# Local import
from core.gabor import MosaicDrawer, SquareDrawer, GaborFilter


class Testscale(unittest.TestCase):

    def setUp(self):
        #  Decide if plotting ok
        self.enable_plotting = True

        # Get gabor bank for texture classification
        self.gabor_bank = GaborFilter(
            ksize=30, n_sigma=4, l_thetas=np.arange(0, np.pi, np.pi / 10), l_lambdas=[5, 10, 15], l_parts=['re', 'im']
        )

        # Set different scale and spatial resolution
        self.scale_1, self.scale_2, self.scale_3, self.resolution_1 = 1, 2, 1.5, 10

        self.sd = SquareDrawer(5, 10, (500, 500), scale=0.05, seed=123, shift=True) \
            .scale_up(self.scale_2 * 3. / 4.)

        self.md = MosaicDrawer((500, 500), 0.9, 10., 0.3, scale=0.05)\
            .scale_up(self.scale_2)

        self.md_2 = MosaicDrawer((500, 500), 0.75, 10., 0.90, scale=0.05)

        self.square_1, self.square_2 = self.sd.image.copy(), self.sd.scaled_image.copy()
        self.mosaic_1, self.mosaic_2 = self.md.image.copy(), self.md.scaled_image.copy()

    def test_scale_texture(self):
        """
        Make sure texture / edge encoding is scale invariant.

        python -m unittest tests.unit.test_scale.Testscale.test_scale_texture
        """
        # Set seed
        random.seed(123)

        # run edge detector on image
        ax_out_1 = np.zeros((self.mosaic_1.shape[0], self.mosaic_1.shape[1], len(self.gabor_bank.keys)))
        ax_out_2 = np.zeros((self.mosaic_2.shape[0], self.mosaic_2.shape[1], len(self.gabor_bank.keys)))
        ax_out_3 = np.zeros((self.md_2.image.shape[0], self.md_2.image.shape[1], len(self.gabor_bank.keys)))

        # Generate random position
        position_x = random.randint(int(self.mosaic_1.shape[1] / 3), int(2 * self.mosaic_1.shape[1] / 3))
        position_y = random.randint(int(self.mosaic_1.shape[0] / 3), int(2 * self.mosaic_1.shape[0] / 3))

        # Encode image with scale 1
        for i, (key, filter) in enumerate(self.gabor_bank.generate_filters(self.scale_1)):
            ax_out_1[:, :, i] = abs(correlate(self.mosaic_1, filter))

        # Encode image with scale 2
        for i, (key, filter) in enumerate(self.gabor_bank.generate_filters(self.scale_2)):
            ax_out_2[:, :, i] = abs(correlate(self.mosaic_2, filter))

        # Encode benchmark image
        for i, (key, filter) in enumerate(self.gabor_bank.generate_filters(self.scale_1)):
            ax_out_3[:, :, i] = abs(correlate(self.md_2.image, filter))

        # Sort scale 1
        l_sorted_indices = np.argsort(ax_out_1[position_x, position_y, :])

        # Build Gabor signature of both scaled image
        ax_out_1 = ax_out_1[position_x, position_y, l_sorted_indices]
        ax_out_1 /= ax_out_1.sum()

        ax_out_2 = ax_out_2[position_x, position_y, l_sorted_indices]
        ax_out_2 /= ax_out_2.sum()

        ax_out_3 = ax_out_3[position_x, position_y, l_sorted_indices]
        ax_out_3 /= ax_out_3.sum()

        self.assertTrue(entropy(ax_out_1, ax_out_3) / entropy(ax_out_1, ax_out_2) > 4)

        if self.enable_plotting:
            # Plot all images
            show_images([self.mosaic_1, self.mosaic_2, self.md_2.image])

            # show encoding bars
            plt.figure(1)
            l_xticks = [self.gabor_bank.keys[i] for i in l_sorted_indices]

            plt.subplot(311)
            plt.bar(range(len(ax_out_1)), ax_out_1)
            plt.xticks(range(len(ax_out_1)), l_xticks)

            plt.subplot(312)
            plt.bar(range(len(ax_out_2)), ax_out_2)
            plt.xticks(range(len(ax_out_2)), l_xticks)

            plt.subplot(313)
            plt.bar(range(len(ax_out_3)), ax_out_3)
            plt.xticks(range(len(ax_out_3)), l_xticks)

            plt.show()

    def test_scale_resolution(self):
        """
        Make sure localisation encoding is scale invariant.

        python -m unittest tests.unit.test_scale.Testscale.test_scale_resolution

        """

        # compute spatial encoding of square 1
        size = int(self.square_1.shape[0] / self.resolution_1)
        spatial_encoding_1 = np.zeros((size, size))

        # Encode image of scale 1
        for i in range(size):
            for j in range(size):
                start_x, end_x = j * self.resolution_1, (j + 1) * self.resolution_1
                start_y, end_y = i * self.resolution_1, (i + 1) * self.resolution_1
                spatial_encoding_1[i, j] = float(self.square_1[start_x:end_x, start_y:end_y].sum()) / size > 0.15

        resolution_2 = int(self.resolution_1 * self.scale_3)
        size_2 = int(self.square_2.shape[0] / resolution_2)
        spatial_encoding_2 = np.zeros((size_2, size_2))

        # Encode image of scale 2
        for i in range(size_2):
            for j in range(size_2):
                start_x, end_x = j * resolution_2, (j + 1) * resolution_2
                start_y, end_y = i * resolution_2, (i + 1) * resolution_2
                spatial_encoding_2[i, j] = float(self.square_2[start_x:end_x, start_y:end_y].sum()) / size > 0.15

        # Compare both image
        missing = (spatial_encoding_1[:size_2, :size_2] - spatial_encoding_2 > 0).sum()
        added = (spatial_encoding_1[:size_2, :size_2] - spatial_encoding_2 < 0).sum()

        # Assert space encoding does not vary too much
        self.assertTrue(missing / spatial_encoding_2.sum() < 0.1)
        self.assertTrue(added / spatial_encoding_2.sum() < 0.25)

        if self.enable_plotting:
            show_images([spatial_encoding_1[:size_2, :size_2], spatial_encoding_2])


def show_images(l_images):
    plt.figure(1)

    for i, img in enumerate(l_images):

        plt.subplot(int('13{}'.format(i + 1)))
        plt.imshow(img, cmap='gray')

    plt.show()