"""All core that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.sparse import spmatrix, csr_matrix, hstack as sphstack
from copy import deepcopy as copy
from functools import lru_cache
import numpy as np

# Local import


@dataclass
class BitMap:
    bf_map: spmatrix
    nb: int
    nf: int
    bf_mask: spmatrix
    basis: np.array
    projections: np.array
    transform_op: np.array

    def __init__(self, bf_map: spmatrix, basis: np.array, ax_transform_op: np.array):
        self.bf_map = bf_map
        self.nb, self.nf = bf_map.shape[0], bf_map.shape[1]
        self.bf_mask = csr_matrix(~bf_map.A)
        self.basis = basis
        self.transform_op = ax_transform_op
        self.projections = self.compute_projections()

    def compute_projections(self) -> np.array:
        return self.basis.dot(self.transform_op)

    def __len__(self):
        return self.bf_map.shape[1]

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
        return self.bf_map[:, i]

    @lru_cache()
    def feature_card(self, n_repeat):
        return self.bf_map.sum(axis=0).A[[0] * n_repeat, :]

    def b2f(self, sax_x):
        return sax_x.T.dot(self.bf_map)

    def f2b(self, sax_x):
        return self.bf_map.dot(sax_x)

    def bitmask(self, sax_x):
        return self.f2b(self.b2f(sax_x).T)


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
            self._meta = [{"id": i} for i in range(len(self.levels))]
            return self._meta

    def set_levels(self, level: int = None):
        self.levels = np.ones(self.inputs.shape[1]) * level
        return self

    @staticmethod
    def empty_comp():
        return FgComponents(csr_matrix((0, 0)), [], np.empty((0,)))

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

    def __getitem__(self, i):
        assert isinstance(i, int), 'index should be an integer'
        return FgComponents(
            inputs=self.inputs[:, i], meta=[self.meta[i]], levels=self.levels[[i]]
        )

    def __add__(self, other):
        if self.inputs.shape[1] == 0:
            sax_inputs = other.inputs
        else:
            sax_inputs = sphstack([self.inputs, other.inputs], format='csr')

        return FgComponents(
            inputs=sax_inputs, meta=self.meta + other.meta,
            levels=np.hstack([self.levels, other.levels])
        )

    def update(self, **kwargs):
        if kwargs.get('meta', None) is not None:
            self.meta = kwargs['meta']

        if kwargs.get('levels', None) is not None:
            self.levels = kwargs['levels']

        if kwargs.get('inputs', None) is not None:
            self.inputs = kwargs['inputs']

        return self

    def copy(self, **kwargs):
        return FgComponents(**{
            'inputs': self.inputs.copy(), 'meta': copy(self.meta),
            'levels': self.levels.copy(), **kwargs
        })

