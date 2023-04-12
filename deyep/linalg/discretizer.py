# Local import
from typing import Union, Tuple, List
from scipy.sparse import csr_matrix, csc_matrix, hstack, spmatrix
import numpy as np

# Global import


class SparsDiscretizer(object):
    def __init__(
            self, n_bin: int, min_val_by_bin: int = 10, quantile_offset: float = 0.02
    ):

        # Core parameters
        self.n_bin = n_bin
        self.min_val_by_bin = min_val_by_bin
        self.quantile_offset = quantile_offset
        self.enc_size = self.n_bin

        # Set unknown attribute to None
        self.bins = None

    def update_enc_size(self) -> None:
        self.enc_size = self.n_bin

    def fit(self, x: np.ndarray) -> 'SparsDiscretizer':

        # Get unique values
        ax_unique = np.unique(x)

        # If not enough unique values, treat it as a cat value and hot encode it
        if len(ax_unique) <= (self.n_bin * self.min_val_by_bin):
            self.bins = ax_unique
            self.n_bin = len(ax_unique)
            self.update_enc_size()
            return self

        # Encode numerical values
        bounds = np.quantile(x[~np.isnan(x)], [self.quantile_offset, 1 - self.quantile_offset])
        self.bins = np.linspace(bounds[0], bounds[1], num=self.n_bin)

        return self

    def transform(self, ax_continuous: np.ndarray) -> spmatrix:

        if self.bins is None:
            raise ValueError('Bins are not set when transform called')

        ax_activation = abs(self.bins - ax_continuous)
        ax_activation = ax_activation == ax_activation.min(axis=1, keepdims=True)

        return csc_matrix(ax_activation)


class ImgCoordDiscretizer:
    def __init__(
            self, n_bin: int, n_directions: int, min_val_by_bin: int = 10, quantile_offset: float = 0.02
    ):
        # Sparse encoder parameters
        self.n_bin = n_bin
        self.min_val_by_bin = min_val_by_bin
        self.quantile_offset = quantile_offset

        # parameters to build bounds
        self.basis = None
        self.augmented_basis = None
        self.y_swaped_basis = None
        self.n_directions = n_directions

        self.transformation = np.vstack([
            -np.sin(np.arange(0, np.pi, np.pi / self.n_directions)),
            np.cos(np.arange(0, np.pi, np.pi / self.n_directions)),
        ])

        #
        self.bf_map = None
        self.n_label = None
        self.encoders = {}

    def fit_transform(self, ax_image: np.ndarray) -> Union[spmatrix, Tuple[spmatrix, spmatrix]]:
        self.fit(ax_image)
        return self.transform(self.augmented_basis, ax_image[self.basis[:, 0], self.basis[:, 1]])

    def fit(self, ax_image: np.ndarray) -> 'ImgCoordDiscretizer':
        # Create basis
        self.basis = np.hstack([
            np.kron(np.arange(ax_image.shape[0]), np.ones(ax_image.shape[1]))[:, np.newaxis],
            np.kron(np.ones(ax_image.shape[0]), np.arange(ax_image.shape[1]))[:, np.newaxis]
        ]).astype(int)

        # Augment basis with more direction
        self.augmented_basis = self.basis.dot(self.transformation)

        # Encode
        self.encode(self.augmented_basis, ax_image[self.basis[:, 0], self.basis[:, 1]])

        return self

    def encode(self, X: np.ndarray, y: np.ndarray) -> 'ImgCoordDiscretizer':
        # Check whether X and y contain only numeric data
        assert X.dtype.kind in set('buifc'), "X contains non num data, must contains only num data"
        assert y.dtype.kind in set('buifc'), "y contains non num data, must contains only num data"

        if self.basis is not None:
            pass

        ax_bf_map, n_inputs = np.zeros((self.n_bin * self.n_directions, self.n_directions), dtype=bool), 0
        for i in range(self.n_directions):
            self.encoders[i] = SparsDiscretizer(self.n_bin, self.min_val_by_bin, self.quantile_offset)\
                .fit(self.augmented_basis[:, i])
            ax_bf_map[range(n_inputs, n_inputs + self.encoders[i].enc_size), i] = True
            n_inputs += self.encoders[i].enc_size

        self.bf_map = csr_matrix(ax_bf_map[:n_inputs, :])
        self.n_label = len(np.unique(y))

        return self

    def transform(
            self, X: np.ndarray, y: np.ndarray = None, transform_input: bool = False
    ) -> Union[spmatrix, Tuple[spmatrix, spmatrix]]:
        assert self.bf_map is not None, "Encoder is not fitted when transform called"

        if transform_input:
            X = X.dot(self.transformation)

        # Encode inputs
        l_encoded = []
        for i in range(self.n_directions):
            l_encoded.append(self.encoders[i].transform(X[:, [i]]))

        if y is not None:
            return hstack(l_encoded), self.transform_labels(y)

        return hstack(l_encoded)

    def transform_labels(self, y: np.ndarray) -> spmatrix:
        # Transform target
        if self.n_label > 2:
            y = csr_matrix(([True] * y.shape[0], (range(y.shape[0]), y)), shape=(y.shape[0], self.n_label))

        else:
            y = csr_matrix(y[:, np.newaxis] > 0)

        return y

    def transform_mask_labels(self, l_masks: List[np.ndarray]):
        return hstack([
            self.transform_labels(ax_img[self.basis[:, 0], self.basis[:, 1]])
            for ax_img in l_masks
        ])


