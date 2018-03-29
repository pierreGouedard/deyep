# Global import
import numpy as npy


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
            max_, min_ = max(max(s - kwargs['treshold']), 0), min(min(s - kwargs['treshold']), 0)

            self.set_bins_from_bound(max_, min_, kwargs['treshold'])

        else:
            raise ValueError('method to compute bins not understood: {}. choose from {}'.format(method,
                                                                                                self.bins_method))

        max_, min_ = max(s), min(s)
        self.set_bins_from_bound(max_, min_)

    def set_bins_from_bound(self, max_, min_, treshold=None):
        delta, r = [0, 0], 0
        if treshold is not None:
            tmp_min, tmp_max = min_, max_
            delta, r = [2 * int(min_ < 0) * treshold, 2 * int(max_ > 0) * treshold], 1

        self.bins = {i: x for i, x in enumerate(npy.linspace(min_ + delta[0], max_ - delta[1], self.k - r))}

        if treshold is not None:
            nn, np, nz = len([x for _, x in self.bins.items() if x < 0]), \
                         len([x for _, x in self.bins.items() if x > 0]), \
                         len([x for _, x in self.bins.items() if x == 0])

            self.bins, offset = {0: 0.0}, 1

            if nn > 0:
                self.bins.update({i + offset: x for i, x in enumerate(npy.linspace(tmp_min, - max(delta), nn + nz))})
                offset = len(self.bins)

            if np > 0:
                self.bins.update({i + offset: x for i, x in enumerate(npy.linspace(max(delta), tmp_max, np + nz))})

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


class Normalizer:

    l_methods = ['variance', 'sum', 'max']

    def __init__(self, method='variance'):

        assert(method in Normalizer.l_methods)
        self.method = method
        self.params = {}

    def set_normalizer(self, s):
        if self.method == 'variance':
            var = npy.var(s)

            if var <=0:
                raise ValueError('semi negative variance computed from signal passed')

            self.params['sigma2'] = var

        elif self.method == 'max':
            max = npy.max(s)

            if max < 0:
                max = npy.min(s)

            if max == 0:
                raise ValueError('max = 0 computed from signal passed')

            self.params['coef'] = max

    def set_transform(self, s):

        # Set normalizer
        self.set_normalizer(s)

        # Transform passed signal
        s_ = self.transform(s)

        return s_

    def transform(self, s):
        if self.method == 'variance':
            s_ = s / self.params['sigma2']
        elif self.method == 'max':
            s_ = s / self.params['max']
        else:
            sum_ = sum(s)
            if sum_ == 0:
                raise ValueError('sum = 0 computed from signal passed')
            s_ = s / sum_

        return s_
