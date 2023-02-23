# Global import
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from PIL import Image
import cv2 as cv

from itertools import product
# Local import


class OpenCVSeq:
    rgb_colors = np.arange(0, 255).astype(np.uint8)
    color_encoding = 'HSV'

    def __init__(self, image, mode: str, seg_coef: float, n_iter: int = 5, color_encoding: str = 'RGB'):
        np.random.seed(12345)
        self.image = image
        self.mode = mode
        self.seg_coef = seg_coef
        self.n_iter = n_iter
        self.color_encoding = color_encoding

         # Load images
        if self.image is None:
            l_images = list(os.listdir(Path(__file__).parent.parent / "data"))
            self.image = random.choice(l_images)

        image = Image.open(Path(__file__).parent.parent / "data" / self.image)
        ax_image = np.asarray(image)
        print(self.image)


        self.plot_img(
            f'segmentation  connected component', ax_image
        )

    @staticmethod
    def plot_img(title, ax_img, ax_target=None):

        if ax_target is not None:
            _, axis = plt.subplots(1, 2, figsize=(20, 10))

            axis[0].imshow(ax_img)
            axis[1].imshow(ax_target)
            plt.title(title)
            plt.show()
        else:
            plt.figure(figsize=(20, 10))
            plt.imshow(ax_img)
            plt.title(title)
            plt.show()

    def segment(self):
        ax_image = np.asarray(self.image)

        # Build origin basis features
        ax_coords = np.hstack([
            np.kron(np.arange(ax_image.shape[0]), np.ones(ax_image.shape[1]))[:, np.newaxis],
            np.kron(np.ones(ax_image.shape[0]), np.arange(ax_image.shape[1]))[:, np.newaxis]
        ]).astype(int)

        if self.mode == 'hierarchical':
            self.segment_hierarchical(ax_image, ax_coords)
        else:
            self.segment_sampling(ax_image, ax_coords, np.ones(ax_image.shape[:-1], dtype=bool))

        self.plot_img(
            f'segmentation  connected component', ax_image
        )

    def segment_hierarchical(self, ax_image: np.ndarray, ax_coords: np.ndarray):

        d_segments = {}
        self.plot_img(f'segmentation / connected component', ax_image)
        # Init
        l_masks = [np.ones(ax_image.shape[:-1], dtype=bool)]
        for i in range(2):
            d_segments[f'level_{i}'] = []
            for j, mask in enumerate(l_masks):
                d_segments[f'level_{i}'].extend(self.segment_sampling(
                    ax_image, ax_coords, mask, comp_min_size=int(0.01 * mask.sum()), display=True
                ))

            l_masks = d_segments[f'level_{i}']
            import IPython
            IPython.embed()
        for ax in l_masks:
            self.plot_img(f'segmentation / connected component', ax)

        import IPython
        IPython.embed()

    def segment_sampling(
            self, ax_image: np.ndarray, ax_coords: np.ndarray, ax_mask: np.ndarray,
            comp_min_size: int = 20, display: bool = True
    ):
        # Init usefull variable
        ax_sub_coord = ax_coords[ax_mask.flatten()]
        tol, i, stop = np.std(ax_image[ax_sub_coord[:, 0], ax_sub_coord[:, 1], :]) * self.seg_coef, 0, False
        l_segs_mask, n_coord = [], ax_mask.sum()

        # Init sample
        ax_coord = ax_sub_coord[np.random.randint(0, int(ax_mask.sum()))]

        # Encode image
        if self.color_encoding == 'HSV':
            # Test less weight to S value
            ax_enc_image = cv.cvtColor(ax_image, cv.COLOR_RGB2HSV)
            ax_enc_image = ax_enc_image * np.array([[[1, 1, 0.5]]])
        elif self.color_encoding == 'HLS':
            ax_enc_image = cv.cvtColor(ax_image, cv.COLOR_RGB2HLS)
        else:
            ax_enc_image = ax_image

        # init arrays
        ax_segment_mask = np.zeros(ax_image.shape[:-1], dtype=bool)
        ax_colored_seg = np.zeros(ax_image.shape, dtype=int) if display else None
        ax_cnt_seg = np.zeros(ax_image.shape[:-1], dtype=int) if display else None

        # Start main loop
        while i < self.n_iter:
            target_val = ax_enc_image[ax_coord[0], ax_coord[1], :]
            ax_norm = norm(ax_enc_image.astype(np.int16) - target_val[np.newaxis, np.newaxis, :], axis=-1)

            # Keep only allowed coordinates
            ax_labels = (ax_norm < tol) * ax_mask
            if (ax_labels * ~ax_segment_mask).sum() < comp_min_size:
                i += 1
                continue

            # Update ax_mask for next coordinate sampling
            l_sub_segs_mask, ax_segments = self.smart_cc(ax_labels, comp_min_size)
            l_segs_mask.extend(l_sub_segs_mask)
            ax_sub_segment_mask = sum(l_sub_segs_mask, np.zeros(ax_segments.shape, dtype=bool))
            ax_segment_mask += ax_sub_segment_mask

            ax_sub_coord_mask = (~ax_segment_mask * ax_mask).flatten()
            if not ax_sub_coord_mask.any():
                break

            # Sample new coord
            ax_coord = ax_coords[ax_sub_coord_mask][np.random.randint(0, int(ax_sub_coord_mask.sum()))]
            ax_segment_mask[ax_coord[0], ax_coord[1]] = True

            print(f'{ax_segment_mask.sum() / n_coord} % segmented')

            if display:
                self.update_seg_display(
                    ax_colored_seg, ax_cnt_seg, ax_segments, ax_sub_segment_mask, ax_image.shape
                )

            i += 1

        # Finally => segment the background:
        if (~ax_segment_mask * ax_mask).sum() > comp_min_size:
            l_sub_segs_mask, ax_segments = self.smart_cc(~ax_segment_mask * ax_mask, comp_min_size)
            l_segs_mask.extend(l_sub_segs_mask)
            if display:
                self.update_seg_display(
                    ax_colored_seg, ax_cnt_seg, ax_segments,
                    sum(l_sub_segs_mask, np.zeros(ax_segments.shape, dtype=bool)), ax_image.shape
                 )

        if display:
            self.plot_img(
                f'segmentation (coef={self.seg_coef}) / connected component',
                (ax_colored_seg / ax_cnt_seg[:, :, np.newaxis]).astype(np.uint8)
            )

        return l_segs_mask

    def smart_cc(self, ax_labels: np.ndarray, comp_min_size: int,):
        # First level connected components
        ret, ax_segments = cv.connectedComponents(ax_labels.astype(np.uint8), connectivity=8)
        lbls_comp, cnt_comp = np.unique(ax_segments, return_counts=True)
        lbls_comp = lbls_comp[1:][cnt_comp[1:] > comp_min_size]
        return [ax_segments == l for l in lbls_comp]

    def update_seg_display(self, ax_colored_seg, ax_cnt_seg, ax_segments, ax_seg_mask, img_shape):
        ax_sub_colors = self.get_random_colors(len(np.unique(ax_segments)))

        # Add colored segment to display image
        ax_colored_seg += ax_sub_colors[(ax_segments * ax_seg_mask).flatten()].reshape((img_shape))
        ax_cnt_seg += ax_seg_mask.astype(int)

    @staticmethod
    def get_random_colors(size: int, black_zeros: bool = True):

        if black_zeros:
            return np.stack([np.zeros(3, dtype=int)] + [
                np.random.choice(OpenCVSeq.rgb_colors, 3) for _ in range(size - 1)
            ])
        else:
            return np.stack([np.random.choice(OpenCVSeq.rgb_colors, 3) for _ in range(size)])


if __name__ == '__main__':
    """
    Usage:
    python deyep/scripts/opencv_seg.py
     
    """
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"

    # Parameters
    image_name = None #''
    mode = 'hierarchical' # 'Nope'
    seg_coef = 1.25
    n_iter = 5
    color_encoding = 'HSV'

    # Load images
    if image_name is None:
        l_images = list(os.listdir(data_path))
        image_name = random.choice(l_images)

    image = Image.open(Path(__file__).parent.parent / "data" / image_name)
    print(f"Segmenting {image_name}")

    opencv_seg = OpenCVSeq(image, mode, seg_coef, n_iter, color_encoding)