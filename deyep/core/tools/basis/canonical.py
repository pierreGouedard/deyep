# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra.natural_domain import get_canonical_basis, get_key_from_series, inner_product, \
    get_canonical_signal
from deyep.utils.names import KVName
from deyep.core.tools.basis.comon import Basis


class CanonicalBasis(Basis):

    def __init__(self, key, N, l_forward_keys, capacity=100):
        Basis.__init__(self, key, N, l_forward_keys, capacity)

    @staticmethod
    def base_from_key(key, offset=0):
        kvkey = KVName.from_string(key)
        N, k = int(kvkey['N']), int(kvkey['k']) + offset
        return get_canonical_basis(N, k)

    @staticmethod
    def signal_from_keys(d_keys):
        N = int(KVName.from_string(d_keys.keys()[0])['N'])
        d_keys = {int(KVName.from_string(k)['k']): v for k, v in d_keys.items()}
        return get_canonical_signal(N, d_keys)

    def keys_from_forward_basis(self, s, n_jobs=0):
        l_keys = get_key_from_series(s)
        return map(lambda x: 'N={},k={}'.format(*x), l_keys)

    def depth_from_basis(self, s, n_jobs=0, return_coef=True):
        l_keys, l_coefs = get_key_from_series(s, set_keys={self.key + i for i in range(self.capacity / 2)},
                                              return_coef=return_coef)
        return map(lambda x: x[1] - self.key, l_keys), l_coefs

    def contain_base(self, s):
        return inner_product(self.base_from_key('k={},N={}'.format(self.key, self.N)), s) != 0
