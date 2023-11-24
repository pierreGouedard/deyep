"""All core that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.sparse import spmatrix, csr_matrix, hstack as sphstack
from copy import deepcopy as copy
from functools import lru_cache
import numpy as np
import random
import string

# Local import


@dataclass
class BitMap:
    """
    util abstraction to store data of an image's sparse representation and provide util operations.

    The sparse representation of an image is a representation where each image pixel is represented as
    discretized and sparsed coordinates in an over complete basis in the 2D euclidean space.

    Each pixel is projected on each direction and the obtained coefficient are discretized so it can take nbit values.
    nbit is user defined but set by default to max(n_row, n_col). Note that if set by default, for the direction
    colinear to the image axis of minimum length, the number of position will be min(n_row, n_col).

    Each coefficient of each direction is transformed to get a sparce indicator boolean matrix of size (1, nbit).
    Each possible discrete coefficients for a particular direction is mapped to a never changing index
    of the sparse indicator boolean matrix. If a pixel p is assigned the discrete coefficient c on direction d,
    then the sparse indicator boolean matrix takes value True at a single index corresponding to c, False elsewhere.

    the sparse indicator boolean matrix is computed for each direction and concatenated so that the sparse
    representation of each pixel is a boolean sparse matrix of shape (1, nbit * n_dir). nbit may be set to
    min(n_row, n_col) for 1 direction as explained earlier.

    Directions are arranged using the anticlockwise direction. If n_dir = 4 the first direction will have coordinates
    (1, 0), 2nd (cos(pi/4), sin(pi/4), ... last (cos(3pi/4), sin(3pi/4)) - wrote using 2D Euclidean usual basis.

    Finally, If an image of size (n_row, n_col) is represented using n_dir directions, it will be represented as a
    sparse boolean matrix of size (n_row*n_col, N) where each row has n_dir
    non false values. Where N = [n_bit * (n_dir - 1)] + min([nbit, n_row, n_col]) and n_bit is lower or equal to
    max(n_row, n_col).

    the stored data are:
        - bitdir_map: sparse array of size (N, n_dir). Column at index i (0<=i<n_dir) have row j = True if it is
          a boolean indicator of direction i's pixel's coefficients
        - nbit: Number of discrete coordinates on each direction.
        - ndir: Number of direction used.
        - bitdir_mask: bitwise complement of bitdir_map.
        - pixel_coords: array of int of shape (n_row*n_col, 2) row i (0<=i<n_row*n_col) gives the coordinates
            (row, col) of pixel in the original image, where row = E[i/n_col] col = (i % n_col) .
        - dir_proj: float arrays of non discretized coefficients of each pixel onto each direction. The array has shape
            (n_row*n_col, n_dir)
        - dir_coords: float arrays representing coordinates of each direction in the original image euclidean basis.
             array has shape (2, n_dir).
    """
    bitdir_map: spmatrix
    nbit: int
    ndir: int
    bitdir_mask: spmatrix
    pixel_coords: np.array
    dir_proj: np.array
    dir_coords: np.array

    def __init__(self, bitdir_map: spmatrix, pixel_coords: np.array, dir_coords: np.array):
        self.bitdir_map = bitdir_map
        self.nbit, self.ndir = bitdir_map.shape[0], bitdir_map.shape[1]
        self.bitdir_mask = csr_matrix(~bitdir_map.A)
        self.pixel_coords = pixel_coords
        self.dir_coords = dir_coords
        self.dir_proj = self.compute_projections()

    def compute_projections(self) -> np.array:
        """
        Compute projections coefficients of each pixel on each directions.

        Returns
        -------

        """
        return self.pixel_coords.dot(self.dir_coords)

    def __len__(self):
        return self.bitdir_map.shape[1]

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self):
            raise StopIteration

        _next = self[self.__idx]
        self.__idx += 1

        return _next

    def __getitem__(self, i):
        assert isinstance(i, int), 'index should be an integer'
        return self.bitdir_map[:, i]

    def norm_ind(self, dir_ind: int) -> int:
        """
        Given a direction index, return the normal direction of that direction.

        the normal direction will always be the next normal direction when
        Parameters
        ----------
        dir_ind

        Returns
        -------

        """
        return (dir_ind + (self.ndir // 2)) % (self.ndir * 2)

    def b2d(self, sax_x: spmatrix) -> spmatrix:
        """
        Count positions of each direction where the passed matrix is non zero.
         The passed sparse matrix has dimension (X, m), where m is an any positive integer. It maps sparse
         representation of pixel to a new space of integer sparse matrix  of dim m).

        The resulting matrix is of size (m, ndir).

        """
        return sax_x.T.dot(self.bitdir_map)

    def d2b(self, sax_x: spmatrix) -> spmatrix:
        """
        Ho god, how to explain that. it is a way to  build the mapping that could be passed to b2d, but here
        all the position corresponding to a direction that is mapped to 1 will be 1

        The resulting matrix is of size (X, m).

        """
        return self.bitdir_map.dot(sax_x)

    def get_sign_dir(self, dir_ind: int) -> int:
        """
        -1 when we go on the other side of the circle.

        Parameters
        ----------
        dir_ind

        Returns
        -------

        """
        # Make sure index dir is correct
        assert 0 <= dir_ind < (self.ndir * 2), f'Index of dir is not in [0, {self.ndir * 2}]'
        return (2 * (dir_ind < self.ndir)) - 1

    def get_proj(self, dir_ind: int, l_sub_inds: Optional[List[int]] = None) -> np.ndarray:
        """
        Get coefficient of each direction and their opposite directions.

        Parameters
        ----------
        dir_ind
        l_sub_inds

        Returns
        -------

        """
        # Make sure ind is between 0 and 2 * nb direction
        dir_ind = dir_ind % (self.ndir * 2)

        # return projections
        if l_sub_inds is not None:
            return self.dir_proj[l_sub_inds, dir_ind % self.ndir] * self.get_sign_dir(dir_ind)

        else:
            return self.dir_proj[:, dir_ind % self.ndir] * self.get_sign_dir(dir_ind)

    def get_dir_coords(self, dir_ind: int) -> np.ndarray:
        """
        Get direction coordinates in usual Euclidean 2D basis. For each direction and their opposite directions. -
        dir_ind in [0, 2 * ndir -1]

        Parameters
        ----------
        dir_ind

        Returns
        -------

        """
        return self.get_sign_dir(dir_ind % (self.ndir * 2)) * self.dir_coords[:, dir_ind % self.ndir]

    def get_norm_coords(self, norm_ind: Optional[int] = None, dir_ind: Optional[int] = None) -> np.ndarray:
        """
        Get normal coordinates in usual Euclidean 2D basis. For each direction and their opposite directions. -
        dir_ind in [0, 2 * ndir -1]

        Parameters
        ----------
        norm_ind
        dir_ind

        Returns
        -------

        """
        assert norm_ind is not None or dir_ind is not None, "At least one of {norm index, dir index} should be set"

        if norm_ind is None:
            norm_ind = self.norm_ind(dir_ind)

        return self.get_sign_dir(norm_ind % (self.ndir * 2)) * self.dir_coords[:, norm_ind % self.ndir]


@dataclass
class FgComponents:
    """Data Structure that store the data necessary to instantiate a firing Graph.

    The data stored are
     * inputs: Input matrix of the firing graph.
     * levels: Levels of the vertices of the firing Graph

    It also provides util function to manipulate firing graph data.

    """
    inputs: spmatrix
    levels: Optional[np.array] = None
    _meta: Optional[List[Dict[str, Any]]] = None

    __idx: int = 0

    @property
    def meta(self):
        if self._meta:
            return self._meta
        else:
            self._meta = [{"id": self.rd_id(5)} for _ in range(len(self.levels))]
            return self._meta

    def set_levels(self, level: int = None):
        self.levels = np.ones(self.inputs.shape[1]) * level
        return self

    @staticmethod
    def empty_comp():
        return FgComponents(csr_matrix((0, 0)), [], [])

    @staticmethod
    def rd_id(n: int = 5):
        return ''.join([random.choice(string.ascii_letters) for _ in range(n)])

    @property
    def empty(self):
        return len(self.levels) == 0

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self):
            raise StopIteration

        _next = self[self.__idx]
        self.__idx += 1

        return _next

    def __len__(self):
        return self.levels.shape[0]

    def __getitem__(self, key):
        if isinstance(key, int):
            ind = key
        elif isinstance(key, str):
            ind = self.get_ind(key)
        else:
            raise TypeError(f'key ({key}) should be string or int')

        return FgComponents(inputs=self.inputs[:, ind], _meta=[self._meta[ind]], levels=self.levels[[ind]])

    def __add__(self, other):
        if self.inputs.shape[1] == 0:
            sax_inputs = other.inputs
        else:
            sax_inputs = sphstack([self.inputs, other.inputs], format='csr')

        return FgComponents(
            inputs=sax_inputs, _meta=self._meta + other._meta,
            levels=np.hstack([self.levels, other.levels])
        )

    def get_ind(self, id: str):
        return [d['id'] for d in self._meta].index(id)

    def set_meta(self, meta: List[Dict[str, Any]]):
        self._meta = meta
        return self

    def update(self, **kwargs):
        if kwargs.get('_meta', None) is not None:
            self._meta = kwargs['_meta']

        if kwargs.get('levels', None) is not None:
            self.levels = kwargs['levels']

        if kwargs.get('inputs', None) is not None:
            self.inputs = kwargs['inputs']

        return self

    def copy(self, **kwargs):
        return FgComponents(**{
            'inputs': self.inputs.copy(), '_meta': copy(self._meta),
            'levels': self.levels.copy(), **kwargs
        })

