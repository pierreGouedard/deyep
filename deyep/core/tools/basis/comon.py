# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra.comon import Upsilon
from deyep.utils.names import KVName


class Basis(object):

    def __init__(self, key, N, l_forward_keys, capacity=20):
        # set base attribute
        self.N = N
        self.key = key
        self.forward_keys = l_forward_keys
        self.queue_forward = [None] * capacity
        self.capacity = capacity

    @property
    def base(self):
        return self.base_from_key('k={},N={}'.format(self.key, self.N))

    @property
    def basis(self):
        return [self.base_from_key('k={},N={}'.format(int(self.key + k), self.N))
                for k in np.arange(0., self.capacity / 2)]

    @property
    def forward_basis(self):
        return [self.base_from_key('k={},N={}'.format(k, self.N)) for k in self.forward_keys]

    @staticmethod
    def from_dict(d_basis):
        return Basis(d_basis.pop('key'), d_basis.pop('N'), d_basis.pop('set_forward_keys'))

    @staticmethod
    def base_from_key(key, offset=0):
        raise NotImplementedError

    @staticmethod
    def signal_from_keys(d_keys):
        raise NotImplementedError

    def keys_from_forward_basis(self, s):
        raise NotImplementedError

    def depth_from_basis(self, s, n_jobs=0):
        raise NotImplementedError

    def contain_base(self, s):
        raise NotImplementedError

    def retrieve_key_from_queue(self, d, t):
        l_keys = [v for k, v in filter(lambda x: x is not None, self.queue_forward) if k == t - (2 * d)]
        if len(l_keys) > 0:
            return set(l_keys[0])

        return set()

    def encode(self, s_forward, timestamp, return_level=False):

        # add incoming key to queue
        l_keys_forward = self.keys_from_forward_basis(s_forward)
        self.queue_forward = [(timestamp, l_keys_forward)] + self.queue_forward[:-1]

        # return base and level if necessay
        if return_level:
            return self.base, len(l_keys_forward)

        return self.base

    def decode(self, s_in, t, keys_input={}, d_levels=None):
        # Make sure s_in contains class instance base
        if not self.contain_base(s_in):
            return None

        l_depths, l_coefs = self.depth_from_basis(s_in)

        # Sum of coef has annihilated
        if len(l_depths) < 2:
            return None

        d_out, s_out = {}, None
        for d, c in zip(l_depths, l_coefs):
            if d == 0:
                continue

            # retrieve forward identifier in queue from keys and build signal.
            set_key_out = self.retrieve_key_from_queue(d, t)
            if len(set_key_out) == 0:
                raise ValueError('Node received signal with depth matching None of the encoding informations')

            if d_levels is not None:
                d_levels[len(set_key_out)] = d_levels.get(len(set_key_out), 0) + Upsilon(c)

            if 2 * (d + 1) < self.capacity:
                for k in set_key_out:
                    d_out[k] = d_out.get(k, 0) + Upsilon(c)
                    if k not in keys_input:
                        k_ = KVName.from_dict({'k': int(KVName.from_string(k)['k']) + d + 1, 'N': self.N}).to_string()
                        d_out[k_] = d_out.get(k_, 0) + Upsilon(c)

        if len(d_out) > 0:
            s_out = self.signal_from_keys(d_out)

        if d_levels is not None and s_out is not None:
            return s_out, d_levels

        return s_out

    @staticmethod
    def from_dict(self, d_basis):
        raise NotImplementedError

    def to_dict(self):
        return {'N': self.N, 'key': self.key, 'forward_keys': self.forward_keys, 'capacity': len(self.queue_forward)}
