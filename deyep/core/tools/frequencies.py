# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra import get_fourrier_coef_from_params, get_fourrier_params, \
    get_fourrier_series, get_fourrier_coef_from_series
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

    def fourrier_basis(self, free=True):
        if free:
            return np.array([self.coef_from_key(k) for k in self.setfree])
        else:
            return np.array([self.coef_from_key(k) for k in self.map.keys()])

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
    def coef_from_series(s, basis, n_jobs=1):
        return get_fourrier_coef_from_series(s, basis, n_jobs=n_jobs)

    @staticmethod
    def key_from_coef(coef):
        N, k = get_fourrier_params(coef)
        return 'N={},k={}'.format(N, k)

    def encode(self, l_coef_in):

        # pop next frequency
        key_out = self.setfree.pop()
        coef_out = FrequencyStack.coef_from_key(key_out)

        # update priorities
        self.priorities[key_out] = self.step
        self.step += 1

        # Update mapping
        self.map.update({key_out: [FrequencyStack.key_from_coef(coef) for coef in l_coef_in]})

        # If set of free frequency is empty make 30% less priority frequency free again
        if len(self.setfree) == 0:
            self.release_key()

        # return signal with poped frequency
        return coef_out

    def decode(self, l_coef_out):

        # pop frequency if it exists in mapping
        l_keys = [self.map[k] for k in map(lambda c: FrequencyStack.key_from_coef(c), l_coef_out) if k is not None]

        if len(l_keys) > 0:
            l_coef_in = sum([[FrequencyStack.coef_from_key(k_) for k_ in k] for k in l_keys], [])
        else:
            l_coef_in = []

        return l_coef_in

    def release_key(self, p=0.3):

        # If set of free frequency is empty renew 30% less priority frequency free again
        l_keys = sorted(self.priorities.items(), key=lambda t: t[1])[:max(int(p * len(self.map)), 1)]

        # Add back less priority freq to set of free frequencies
        self.setfree = self.setfree.union(set([t[0] for t in l_keys]))

    def to_dict(self):
        return {'N': self.N, 'l_keys': self.l_keys, 'setfree': self.setfree, 'map': self.map,
                'priorities': self.priorities, 'step': self.step}
