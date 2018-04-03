# Global import
import numpy as npy
from scipy.sparse import lil_matrix

class Discretizer:
    bins_method = ['bounds', 'signal', 'treshold']

    def __init__(self, k, method='bins', bins=None):
        self.method = method
        self.k = k
        self.bins = bins
        self.params = None

    def set_discretizer_bins(self, s=None, method='bounds', **kwargs):

        if method == 'bounds':
            self.bins = {i: x for i, x in enumerate(npy.linspace(kwargs['min'], kwargs['max'], self.k))}

        elif method == 'signal':
            max_, min_ = max(s), min(s)
            self.set_bins_from_bound(max_, min_)

        elif method == 'treshold':
            max_, min_ = max(max(s - 2 * kwargs['treshold']), 0), min(min(s + 2 * kwargs['treshold']), 0)
            self.set_bins_from_bound(max_, min_, kwargs['treshold'])

        else:
            raise ValueError('method to compute bins not understood: {}. choose from {}'.format(method,
                                                                                                self.bins_method))

    def set_bins_from_bound(self, max_, min_, treshold=None):

        # Computes bins
        r = int(treshold is not None)
        self.bins = {i: x for i, x in enumerate(npy.linspace(min_, max_, self.k - r))}

        if treshold is not None:
            nn, np, nz = len([x for _, x in self.bins.items() if x < 0]), \
                         len([x for _, x in self.bins.items() if x > 0]), \
                         len([x for _, x in self.bins.items() if x == 0])

            self.bins, offset = {0: 0.0}, 1

            if nn > 0:
                self.bins.update({i + offset: x for i, x in enumerate(npy.linspace(min_, - 2 * treshold, nn + nz))})
                offset = len(self.bins)

            if np > 0:
                self.bins.update({i + offset: x for i, x in enumerate(npy.linspace(2 * treshold, max_, np + nz))})

    def discretize_value(self, x):

        if self.bins is None:
            raise ValueError('First set the bins of discretizer')

        x_ = min([(v, abs(x - v)) for k, v in self.bins.items()], key=lambda t: t[1])[0]

        return x_

    def discretize_array(self, ax):

        # Vectorize discretie value function
        vdicretizer = npy.vectorize(lambda x: self.discretize_value(x))

        # Apply to array
        ax_ = vdicretizer(ax)

        return ax_

    def encode_1d_array(self, ax, sparse=False):

        ax_out = npy.zeros(len(ax) * self.k)
        ax_ = self.discretize_array(ax)

        for i, x in enumerate(ax_):
            ax_out[i * self.k: (i + 1) * self.k] = \
                npy.array([b == x for _, b in sorted(self.bins.items(), key=lambda t: t[0])])

        if sparse:
            ax_out = lil_matrix(ax_out)

        return ax_out

    def encode_2d_array(self, ax, sparse=False, orient='lines'):
        if orient == 'columns':
            ax = ax.transpose()

        ax_out = npy.zeros([len(ax), len(ax[0]) * self.k])

        for i, ax_ in enumerate(ax):
            ax_out[i, :] = self.encode_1d_array(ax_)

        if orient == 'columns':
            ax_out = ax_out.transpose()

        if sparse:
            ax_out = lil_matrix(ax_out)

        return ax_out

    def decode_1d_array(self, ax, sparse=False):

        if sparse:
            ax = ax.toarray()[0]

        ax_out = npy.zeros(int(len(ax) / self.k))

        for i in range(len(ax_out)):
            ax_out[i] = self.bins.get(npy.where(ax[i * self.k: (i + 1) * self.k])[0][0])

        return ax_out

    def decode_2d_array(self, ax, sparse=False, orient='lines'):

        if orient == 'columns':
            ax = ax.transpose()

        if sparse:
            ax = ax.toarray()

        ax_out = npy.zeros([len(ax), int(len(ax[0]) / self.k)])

        for i, ax_ in enumerate(ax):
            ax_out[i, :] = self.decode_1d_array(ax_)

        if orient == 'columns':
            ax_out = ax_out.transpose()

        return ax_out


class Normalizer:

    l_methods = ['variance', 'sum', 'max']

    def __init__(self, method='variance'):

        assert(method in Normalizer.l_methods)
        self.method = method
        self.params = {}

    def set_normalizer(self, s):
        if self.method == 'variance':
            var = npy.var(s)

            if var <= 0:
                raise ValueError('semi negative variance computed from signal passed')

            self.params['sigma'] = npy.sqrt(var)

        elif self.method == 'max':
            max = npy.max(s)

            if max < 0:
                max = - npy.min(s)

            if max == 0:
                raise ValueError('max = 0 computed from signal passed')

            self.params['max'] = max

    def set_transform(self, s):

        # Set normalizer
        self.set_normalizer(s)

        # Transform passed signal
        s_ = self.transform(s)

        return s_

    def transform(self, s):
        if self.method == 'variance':
            s_ = s / self.params['sigma']
        elif self.method == 'max':
            s_ = s / self.params['max']
        else:
            sum_ = sum(s)
            if sum_ == 0:
                raise ValueError('sum = 0 computed from signal passed')
            s_ = s / sum_

        return s_
