# Global import
import numpy as np

# Local import
from deyep.core.tools.linear_algebra import get_fourrier_coef, get_fourrier_key, get_fourrier_series
from deyep.utils.names import KVName


class FrequencyStack(object):

    def __init__(self, size, key, capacity, setfree=None, priorities=None, map=None, step=None):
        # set base attribute
        self.N = size
        self.k_ = key
        self.capacity = capacity
        self.setfree = FrequencyStack.init_setfree(self.N, self.capacity, self.k_) if setfree is None else setfree
        self.step = 0 if step is None else step
        self.priorities = dict() if priorities is None else priorities
        self.map = dict() if map is None else map

    @property
    def basis(self):
        return np.array(get_fourrier_coef(self.N, self.k_ + i) for i in range(self.capacity))

    @callable
    def signal(self, coef):
        return get_fourrier_series(coef)

    @staticmethod
    def from_dict(d_frequency_stack):
        return FrequencyStack(d_frequency_stack.pop('N'), d_frequency_stack.pop('k_'), d_frequency_stack.pop('capacity'),
                              **d_frequency_stack)

    @staticmethod
    def init_setfree(N, c, k):
        return {'N={},k={}'.format(N, k + i) for i in range(c)}

    @staticmethod
    def coef_from_str(key):
        N, k = KVName.from_string(key)['N'], KVName.from_string(key)['k']
        return get_fourrier_coef(N, k)

    @staticmethod
    def str_from_coef(freq):
        N, k = get_fourrier_key(freq)
        return 'N={},k={}'.format(N, k)

    def encode(self, coef_in):

        # pop next frequency
        key_out = self.setfree.pop()
        coef_out = FrequencyStack.coef_from_str(key_out)

        # update priorities
        self.priorities[key_out] = self.step
        self.step += 1

        # Update mapping
        self.map.update({key_out: FrequencyStack.str_from_coef(coef_in)})

        # If set of free frequency is empty make 30% less priority frequency free again
        self.release_key()

        # return signal with poped frequency
        return coef_out

    def decode(self, coef_out):

        # pop frequency if it exists in mapping
        key_ = self.map.get(FrequencyStack.str_from_coef(coef_out), None)

        if key_ is not None:
            coef_ = FrequencyStack.coef_from_str(key_)
        else:
            coef_ = 0

        return coef_

    def release_key(self):

        # If set of free frequency is empty renew 30% less priority frequency free again
        l_keys = sorted(self.priorities.items(), key=lambda t: t[1])[:int(0.3 * len(self.map))]

        # Add back less priority freq to set of free frequencies
        self.setfree = self.setfree.union(set([t[0] for t in l_keys]))

    def to_dict(self):
        return {'N': self.N, 'k_': self.k_, 'capacity': self.capacity, 'setfree': self.setfree, 'map': self.map,
                'priorities': self.priorities, 'step': self.step}
