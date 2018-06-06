# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra import get_fourrier_coef_from_params, get_fourrier_params, \
    get_fourrier_series, get_fourrier_coef_from_series, Upsilon
from deyep.utils.names import KVName


class FrequencyStack(object):

    def __init__(self, size, l_keys, setfree=None, priorities=None, map=None, step=None):
        # set base attribute
        self.N = size
        self.l_keys = l_keys
        self.setfree = FrequencyStack.init_setfree(l_keys, self.N) if setfree is None else setfree
        self.step = 0 if step is None else step
        self.priorities = dict() if priorities is None else priorities
        self.map = dict() if map is None else map

    @property
    def basis_specific(self):
        return [self.coef_from_key('k={},N={}'.format(k, self.N)) for k in self.l_keys]

    @property
    def basis_generic(self):
        return [self.coef_from_key('k={},N={}'.format(k, self.N)) for k in range(self.N / 2)]

    def fourrier_basis(self, free=True):
        if free:
            return np.array([self.series_from_coef(self.coef_from_key(k)) for k in self.setfree])
        else:
            return np.array([self.series_from_coef(self.coef_from_key(k)) for k in self.map.keys()])

    @staticmethod
    def from_dict(d_frequency_stack):
        return FrequencyStack(d_frequency_stack.pop('N'), d_frequency_stack.pop('l_keys'), **d_frequency_stack)

    @staticmethod
    def init_setfree(l_keys, N):
        return {'N={},k={}'.format(N, k) for k in l_keys}

    @staticmethod
    def coef_from_key(key):
        N, k = int(KVName.from_string(key)['N']), int(KVName.from_string(key)['k'])
        return get_fourrier_coef_from_params(N, k)

    @staticmethod
    def series_from_coef(coef):
        return get_fourrier_series(coef)

    @staticmethod
    def coef_from_series(s, basis, n_jobs=1, return_coef=False):
        return get_fourrier_coef_from_series(s, basis, n_jobs=n_jobs, return_coef=return_coef)

    @staticmethod
    def key_from_coef(coef):
        N, k = get_fourrier_params(coef)
        return 'N={},k={}'.format(N, k)

    def encode(self, s_in, return_level=False):

        # pop next frequency
        key_out = self.setfree.pop()
        s_out = self.series_from_coef(FrequencyStack.coef_from_key(key_out))

        # update priorities
        self.priorities[key_out] = self.step
        self.step += 1

        # Update mapping
        l_coef_in = self.coef_from_series(s_in, self.basis_generic, n_jobs=0)
        self.map.update({key_out: [FrequencyStack.key_from_coef(coef) for coef in l_coef_in]})

        # If set of free frequency is empty make 30% less priority frequency free again
        if len(self.setfree) == 0:
            self.release_key()

        # return signal with poped frequency and level if necessay
        if return_level:
            return s_out, len(l_coef_in)

        return s_out

    def decode(self, s_in):

        # pop frequency if it exists in mapping
        l_coefs_in, l_coefs = self.coef_from_series(s_in, [self.coef_from_key(k) for k in self.map.keys()], n_jobs=0,
                                                    return_coef=True)
        s_out = np.zeros(self.N)
        for i, c_in in enumerate(l_coefs_in):
            l_coef_out = map(lambda x: self.coef_from_key(x), self.map[self.key_from_coef(c_in)])
            for c_out in l_coef_out:
                s_out = Upsilon(l_coefs[i]) * self.series_from_coef(c_out)

        return s_out

    def release_key(self, p=0.3):

        # If set of free frequency is empty renew 30% less priority frequency free again
        l_keys = sorted(self.priorities.items(), key=lambda t: t[1])[:max(int(p * len(self.map)), 1)]

        # Add back less priority freq to set of free frequencies
        self.setfree = self.setfree.union(set([t[0] for t in l_keys]))

    def to_dict(self):
        return {'N': self.N, 'l_keys': self.l_keys, 'setfree': self.setfree, 'map': self.map,
                'priorities': self.priorities, 'step': self.step}
