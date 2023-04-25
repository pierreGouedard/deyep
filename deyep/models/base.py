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
        return (dir_ind + (self.ndir // 2)) % (self.ndir * 2)

    def b2d(self, sax_x: spmatrix) -> spmatrix:
        return sax_x.T.dot(self.bitdir_map)

    def d2b(self, sax_x: spmatrix) -> spmatrix:
        return self.bitdir_map.dot(sax_x)

    def get_coef_dir(self, dir_ind: int) -> int:
        # Make sure index dir is correct
        assert 0 <= dir_ind < (self.ndir * 2), f'Index of dir is not in [0, {self.ndir * 2}]'
        return (2 * (dir_ind < self.ndir)) - 1

    def get_proj(self, dir_ind: int, l_sub_inds: Optional[List[int]] = None) -> np.ndarray:

        # Make sure ind is between 0 and 2 * nb direction
        dir_ind = dir_ind % (self.ndir * 2)

        # return projections
        if l_sub_inds is not None:
            return self.dir_proj[l_sub_inds, dir_ind % self.ndir] * self.get_coef_dir(dir_ind)

        else:
            return self.dir_proj[:, dir_ind % self.ndir] * self.get_coef_dir(dir_ind)

    def get_dir_coords(self, dir_ind: int) -> np.ndarray:
        return self.get_coef_dir(dir_ind % (self.ndir * 2)) * self.dir_coords[:, dir_ind % self.ndir]

    def get_norm_coords(self, norm_ind: Optional[int] = None, dir_ind: Optional[int] = None) -> np.ndarray:
        assert norm_ind is not None or dir_ind is not None, "At least one of {norm index, dir index} should be set"

        if norm_ind is None:
            norm_ind = self.norm_ind(dir_ind)

        return self.get_coef_dir(norm_ind % (self.ndir * 2)) * self.dir_coords[:, norm_ind % self.ndir]


@dataclass
class FgComponents:
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

