# Local import
from typing import Union, Tuple, List
from scipy.sparse import csr_matrix, csc_matrix, hstack, spmatrix
import numpy as np

# Global import


class SparsDiscretizer(object):
    """
    Discretize float values and build a sparse representation.

    Given a list of n float samples that are discretized into n_bin values, the sparse representation
    has a shape of (n, n_bin), where each row has only 1 non nonzero value that correspond to a unique bin
    activated by index corresponding sample.

    """
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
        """
        Update number of bin used to discretize values.

        Returns
        -------

        """
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
    """
    Transform image pixel into a discretized & sparse overcomplete representation.
    """
    def __init__(
            self, n_bin: int, n_directions: int, min_val_by_bin: int = 10, quantile_offset: float = 0.02
    ):
        # Sparse encoder parameters
        self.n_bin = n_bin
        self.min_val_by_bin = min_val_by_bin
        self.quantile_offset = quantile_offset

        # parameters to build bounds
        self.pixel_coords = None
        self.augmented_coords = None
        self.n_directions = n_directions

        # Create overcomplete basis of size (2, n_dir), transformation[0, :] are y axis coef and
        # transformation[1, :] are x axis coef (in the usual 2D Euclidean space)
        self.transformation = np.vstack([
            -np.sin(np.arange(0, np.pi, np.pi / self.n_directions)),
            np.cos(np.arange(0, np.pi, np.pi / self.n_directions)),
        ])

        # Init discretisation metadata
        self.bitdir_map = None
        self.encoders = {}

    def fit_transform(self, ax_image: np.ndarray) -> Union[spmatrix, Tuple[spmatrix, spmatrix]]:
        """
        Fit discretizer and transform image.

        Parameters
        ----------
        ax_image: Array of bool (binary image).

        Returns
        -------

        """
        self.fit(ax_image)
        return self.transform(self.augmented_coords, ax_image[self.pixel_coords[:, 0], self.pixel_coords[:, 1]])

    def fit(self, ax_image: np.ndarray) -> 'ImgCoordDiscretizer':
        """
        Fit discretisation & sparsification of passed boolean image

        Parameters
        ----------
        ax_image: Array of bool (binary image of shape (nrow, ncol).


        Returns
        -------

        """
        # Create img coords, shape is (nrow*ncol, 2), row i (0<=i<nrow*ncol) gives the coordinates
        # (row, col) of pixel in the original image, where row = E[i/n_col] & col = (i % n_col).
        # in term of usual 2D euclidean space, the first columns store y axis values and second x axis values.
        self.pixel_coords = np.hstack([
            np.kron(np.arange(ax_image.shape[0]), np.ones(ax_image.shape[1]))[:, np.newaxis],
            np.kron(np.ones(ax_image.shape[0]), np.arange(ax_image.shape[1]))[:, np.newaxis]
        ]).astype(int)

        # Augment coords with more direction. it computes the orthogonal projection of each pixel onto
        # each direction of the overcomplet basis. The shape of the result is (nrow*n_col, n_dir)
        self.augmented_coords = self.pixel_coords.dot(self.transformation)

        # Represent augmented coords as a sparse matrix
        self.encode(self.augmented_coords)

        return self

    def encode(self, X: np.ndarray) -> 'ImgCoordDiscretizer':
        """
        Fit discretizer a matrix of numeric values.

        Let X, the input matrix have a shape (n, m) with numeric values. Each row of the matrix correspond to
        a particular position of a pixel in an image. Each column correpond to the orthogonal projection of the
        pixel's position on a particular direction in the 2D euclidean space.

        For each direction we discretize the projections coefficient of each sample using nbit bins.

        Discrete projection coefficient may then be transformed to get a sparse indicator boolean matrix
        of size (n, nbit) for each direction. Concatenation of each sparse representation of discretized
        projection coefficient would provide a discretized & sparsed representation of shape (n, nbit * m).

        In addition, a sparse array of size (nbit * m , m) is built. It enables to keep track of indices in
        transformed matrix used for each direction's sparse representation. The Column index i (0<=i<m) has row v
        alue set to True at index used for sparse representation of discretize projection coefficient of direction i,
        False elsewhere.

        Parameters
        ----------
        X: input array to transform.

        Returns
        -------

        """
        # Check whether X and y contain only numeric data
        assert X.dtype.kind in set('buifc'), "X contains non num data, must contains only num data"

        if self.pixel_coords is not None:
            pass

        # For each direction, discretize projection coefficient and build its sparse representation.
        ax_bd_map, n_inputs = np.zeros((self.n_bin * self.n_directions, self.n_directions), dtype=bool), 0
        for i in range(self.n_directions):
            self.encoders[i] = SparsDiscretizer(self.n_bin, self.min_val_by_bin, self.quantile_offset)\
                .fit(self.augmented_coords[:, i])
            ax_bd_map[range(n_inputs, n_inputs + self.encoders[i].enc_size), i] = True
            n_inputs += self.encoders[i].enc_size

        self.bitdir_map = csr_matrix(ax_bd_map[:n_inputs, :])

        return self

    def transform(
            self, X: np.ndarray, transform_input: bool = False
    ) -> Union[spmatrix, Tuple[spmatrix, spmatrix]]:
        """
        Transform a matrix of numeric values into a discrtetized sparse matrice from fitted discretizer.

        Let X, the input matrix have a shape (n, m) with numeric values. Each row of the matrix correspond to
        the position of a pixel in an image. Each column correpond to the orthogonal projection of the pixel
        position of a particular direction in the 2D euclidean space.

        For each direction, we fit a discretizer on every projections coefficients of each sample using nbit bins.

        For each direction, discrete projection coefficient are transformed to get a sparse indicator boolean matrix
        of size (n, nbit). Each possible discrete coefficients for a particular direction is mapped to a never
        changing column index of the sparse indicator boolean matrix.

        If a pixel p is assigned the discrete coefficient c on direction d, then the sparse indicator boolean matrix
        takes value True at a single column index corresponding to c, False elsewhere.

        the sparse indicator boolean matrix computed for each direction are concatenated so that the sparse
        representation of the discretized matrix is transformed as a boolean sparse matrix of shape
        (1, nbit * m).

        Parameters
        ----------
        X: input array to transform.
        transform_input: boolean that indicate whether direction augmentation should be applied to X prior to
            Discretization and sparsification transform.

        Returns
        -------

        """
        assert self.bitdir_map is not None, "Encoder is not fitted when transform called"

        if transform_input:
            X = X.dot(self.transformation)

        # Encode inputs
        l_encoded = []
        for i in range(self.n_directions):
            l_encoded.append(self.encoders[i].transform(X[:, [i]]))

        return hstack(l_encoded)

    def transform_mask_labels(self, l_masks: List[np.ndarray]):
        """
        Reshape list of boolean images.

        Given n mask (boolean image) of size (nrow, ncol), it build a boolean sparse matrix of shape (n_row*n_col, n)
        the value at position (i, j), i in [0, nrow*ncol[, j in [0, n[ is equalto the value of mask l_masks[j] at
        position row = E[i/ncol] col = (i % ncol).

        Parameters
        ----------
        l_masks: list of boolean image.

        Returns
        -------
        Reshaped sparse masks.
        """
        l_coord_masks = [
            ax_mask[self.pixel_coords[:, 0], self.pixel_coords[:, 1]] for ax_mask in l_masks
        ]
        return hstack([csr_matrix(ax_coord_mask[:, np.newaxis] > 0) for ax_coord_mask in l_coord_masks])
