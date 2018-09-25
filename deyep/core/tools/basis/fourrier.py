# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra.fourrier_domain import get_fourrier_series, get_fourrier_basis_from_series
from deyep.utils.names import KVName
from deyep.core.tools.basis.comon import Basis


class FourrierBasis(Basis):

    def __init__(self, key, N, l_forward_keys, capacity=100):
        Basis.__init__(self, key, N, l_forward_keys, capacity)

    @staticmethod
    def base_from_key(key, offset=0):
        N, k = int(KVName.from_string(key)['N']), int(KVName.from_string(key)['k']) + offset
        return get_fourrier_series(N, k)

    @staticmethod
    def signal_from_keys(d_keys):
        return sum([v * FourrierBasis.base_from_key(k) for k, v in d_keys.items()])

    def keys_from_forward_basis(self, s):
        l_indices = get_fourrier_basis_from_series(s, np.array(self.forward_basis, dtype=complex))
        return map(lambda x: 'N={},k={}'.format(self.N, self.forward_keys[x]), l_indices)

    def depth_from_basis(self, s, return_coef=True):
        l_indices, l_coefs = get_fourrier_basis_from_series(s, np.array(self.basis, dtype=complex),
                                                            return_coef=return_coef)
        return l_indices, l_coefs

    def contain_base(self, s):
        return np.round(np.real(self.base_from_key('k={},N={}'.format(self.key, self.N)).dot(s.conjugate()))) != 0.

    @staticmethod
    def from_dict(d_basis):
        return FourrierBasis(d_basis['key'], d_basis['N'], d_basis['forward_keys'], d_basis['capacity'])
