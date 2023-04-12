# Global import
import logging
import numpy as np
from numpy.linalg import norm
from PIL import Image
import cv2 as cv

# Local import
from deyep.utils.plots import plot_img
from deyep.opencv.utils import get_mask_cc


class OpenCVSeq:
    # TODO: I htink more level is not necessary, but different encoding of the image RBG HSV + multiple coef seems
    #   enough => Next is encoding shapes !!!
    rgb_colors = np.arange(0, 255).astype(np.uint8)
    color_encoding = 'HSV'

    def __init__(
            self, image: Image, seg_coef: float, n_iter: int = 5, color_encoding: str = 'RGB',
            color_weights: np.array = np.ones(3), depth: int = 2, logger: logging.Logger = None
    ):
        np.random.seed(12345)

        # Image parameters
        self.image = np.asarray(image)
        self.pixel_shape = self.image.shape[:-1]
        self.color_encoding = color_encoding
        self.color_weights = color_weights

        self.encode_image()

        # Segmentation parameters
        self.seg_coef = seg_coef
        self.n_iter = n_iter
        self.depth = depth

        # properties
        self._coords = None

        # Other
        self.logger = logger or logging.getLogger('Opencv - segment')


    @property
    def coords(self):
        if self._coords is None:
            self._coords = np.hstack([
                np.kron(np.arange(self.image.shape[0]), np.ones(self.image.shape[1]))[:, np.newaxis],
                np.kron(np.ones(self.image.shape[0]), np.arange(self.image.shape[1]))[:, np.newaxis]
            ]).astype(int)

            return self._coords

        return self._coords

    def show_img(self, title, ax_target=None):
        plot_img(title, np.asarray(self.image), ax_target)

    def encode_image(self):
        if self.color_encoding == 'HSV':
            # Test less weight to S value
            self.image = cv.cvtColor(self.image, cv.COLOR_RGB2HSV)
            self.image = (self.image * self.color_weights).astype(np.uint8)
        elif self.color_encoding == 'HLS':
            self.image = cv.cvtColor(self.image, cv.COLOR_RGB2HLS)
            self.image = (self.image * self.color_weights).astype(np.uint8)
        else:
            self.image = (self.image * self.color_weights).astype(np.uint8)

    def segment(self):

        d_segments = {}
        self.show_img(f'segmentation / connected component')

        # Init
        l_masks = [np.ones(self.pixel_shape, dtype=bool)]
        for i in range(self.depth):
            d_segments[f'depth_{i}'] = []
            for j, mask in enumerate(l_masks):
                d_segments[f'depth_{i}'].extend(self.segment_sampling(
                    self.image, self.coords, mask, comp_min_size=int(0.01 * mask.sum()), display=True
                ))

            l_masks = d_segments[f'depth_{i}']

        for ax in l_masks:
            self.show_img(f'segmentation / connected component', ax)

        import IPython
        IPython.embed()

    def segment_sampling(
            self, ax_image: np.ndarray, ax_coords: np.ndarray, ax_mask: np.ndarray,
            comp_min_size: int = 20, display: bool = True
    ):
        # Init usefull variable
        ax_sub_coords = ax_coords[ax_mask.flatten()]

        # Infere tolerance
        tol = np.std(ax_image[ax_sub_coords[:, 0], ax_sub_coords[:, 1], :]) * self.seg_coef

        # Init arrays
        ax_comp_mask, ax_bin = ax_mask.copy(), np.zeros(self.pixel_shape, dtype=bool)

        # Start main loop
        l_comp_mask, i, n_coord = [], 0, ax_mask.sum()
        while i < self.n_iter:
            # If nothing to sample stop.
            if not ax_comp_mask.any():
                break

            # Sample coordinate
            ax_coord = ax_coords[ax_comp_mask.flatten()][np.random.randint(0, int(ax_comp_mask.sum()))]
            ax_comp_mask[ax_coord[0], ax_coord[1]] = False
            target_val = ax_image[ax_coord[0], ax_coord[1], :]

            # Compute norm with other pixel image
            ax_flat_norm = norm(
                ax_image[ax_sub_coords[:, 0], ax_sub_coords[:, 1], :].astype(np.int16) -
                target_val[np.newaxis, np.newaxis, :], axis=-1
            )
            # Keep only close enough pixel
            ax_flat_labels = (ax_flat_norm < tol)
            if ax_flat_labels.sum() < comp_min_size:
                i += 1
                continue

            # Get connected components
            ax_bin[ax_sub_coords[:, 0], ax_sub_coords[:, 1]] = ax_flat_labels
            l_sub_comp_mask = get_mask_cc(ax_bin.astype(np.uint8), comp_min_size)
            ax_bin[:, :] = False

            # Get connected components
            l_comp_mask.extend(l_sub_comp_mask)

            # Update segment mask
            ax_comp_mask *= ~sum(l_sub_comp_mask, np.zeros(self.pixel_shape, dtype=bool))

            self.logger.info(f'{round(((n_coord - ax_comp_mask.sum()) / n_coord) * 100, 3)}  % segmented')
            i += 1

        # Finally => segment the background:
        if ax_comp_mask.sum() > comp_min_size:
            l_sub_segs_mask = get_mask_cc(ax_comp_mask.astype(np.uint8), comp_min_size)
            l_comp_mask.extend(l_sub_segs_mask)

        return l_comp_mask

    @staticmethod
    def get_random_colors(size: int, black_zeros: bool = False):

        if black_zeros:
            return np.stack([np.zeros(3, dtype=int)] + [
                np.random.choice(OpenCVSeq.rgb_colors, 3) for _ in range(size - 1)
            ])
        else:
            return np.stack([np.random.choice(OpenCVSeq.rgb_colors, 3) for _ in range(size)])
