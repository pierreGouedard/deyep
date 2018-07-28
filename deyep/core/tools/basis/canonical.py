# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra.natural_domain import get_canonical_basis, get_key_from_series, \
    get_fourrier_series, get_fourrier_coef_from_series, inner_product
from deyep.utils.names import KVName
from deyep.core.tools.basis.comon import Basis


class CanonicalBasis(Basis):

    def __init__(self, key, N, l_forward_keys, capacity=100):
        Basis.__init__(self, key, N, l_forward_keys, capacity)

    @staticmethod
    def base_from_key(key, offset=0):
        N, k = int(KVName.from_string(key)['N']), int(KVName.from_string(key)['k']) + offset
        return get_canonical_basis(N, k)

    def keys_from_forward_basis(self, s, n_jobs=0):
        l_keys = get_key_from_series(s)
        return map(lambda x: 'N={},k={}'.format(*x), l_keys)

    def depth_from_basis(self, s, n_jobs=0, return_coef=False):
        l_keys, l_coefs = get_key_from_series(s, set_keys={i for self.key + i in range(self.capacity)},
                                              return_coef=return_coef)
        return map(lambda x: int(KVName.from_string(x)['k']) - self.key, l_keys), l_coefs

    def contain_base(self, s):
        return inner_product(self.base_from_key(self.key), s) != 0

