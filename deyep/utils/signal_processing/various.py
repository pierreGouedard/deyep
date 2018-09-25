import numpy as np
from scipy.sparse import csc_matrix


class Discretizer:
    bins_method = ['bounds', 'signal', 'signal_clustering', 'treshold']

    def __init__(self, k, method='bins', bins=None):
        self.method = method
        self.k = k
        self.bins = bins
        self.params = None

    def set_discretizer_bins(self, s=None, method='bounds', **kwargs):

        if method == 'bounds':
            self.bins = {i: x for i, x in enumerate(np.linspace(kwargs['min'], kwargs['max'], self.k))}

        elif method == 'signal':
            max_, min_ = max(s), min(s)
            self.set_bins_from_bound(max_, min_)

        elif method == 'signal_clustering':
            n = kwargs['n_cluster']
            l_clusters = [np.percentile(s, 100 * (float(i) / n)) for i in range(n + 1)]
            self.set_bins_from_clusters(l_clusters, self.k / n)

        elif method == 'treshold':
            max_, min_ = max(max(s - 2 * kwargs['treshold']), 0), min(min(s + 2 * kwargs['treshold']), 0)
            self.set_bins_from_bound(max_, min_, treshold=kwargs['treshold'])

        else:
            raise ValueError('method to compute bins not understood: {}. choose from {}'.format(method,
                                                                                                self.bins_method))
        return self

    def set_bins_from_bound(self, max_, min_, treshold=None, res=1e-4):

        # Computes bins
        r = int(treshold is not None)

        if max_ - res < min_:
            max_ = min_

        self.bins = {i: x for i, x in enumerate(np.unique(np.linspace(min_, max_, self.k - r)))}

        if treshold is not None:
            nneg, npos, nz = len([x for _, x in self.bins.items() if x < 0]), \
                             len([x for _, x in self.bins.items() if x > 0]), \
                             len([x for _, x in self.bins.items() if x == 0])

            self.bins, offset = {0: 0.0}, 1

            if nneg > 0:
                self.bins.update({i + offset: x for i, x in enumerate(np.linspace(min_, - 2 * treshold, nneg + nz))})
                offset = len(self.bins)

            if npos > 0:
                self.bins.update({i + offset: x for i, x in enumerate(np.linspace(2 * treshold, max_, npos + nz))})

    def set_bins_from_clusters(self, l_clusters, k_):
        self.bins = {}
        for i, (min_, max_) in enumerate(zip(l_clusters[:-1], l_clusters[1:])):
            self.bins.update({i * k_ + j: x for j, x in enumerate(np.unique(np.linspace(min_, max_, k_ + 1))[:-1])})

    def discretize_value(self, x):

        if self.bins is None:
            raise ValueError('First set the bins of discretizer')

        x_ = min([(v, abs(x - v)) for k, v in self.bins.items()], key=lambda t: t[1])[0]

        return x_

    def discretize_array(self, ax):

        # Vectorize discretie value function
        vdicretizer = np.vectorize(lambda x: self.discretize_value(x))

        # Apply to array
        ax_ = vdicretizer(ax)

        return ax_

    def arange(self, x_min, x_max):
        x_min_, x_max_ = self.discretize_value(x_min), self.discretize_value(x_max)
        return np.array(sorted([v for _, v in self.bins.items() if x_min_ <= v <= x_max_]))

    def encode_1d_array(self, ax, sparse=False):

        ax_out = np.zeros(len(ax) * self.k)
        ax_ = self.discretize_array(ax)

        for i, x in enumerate(ax_):
            ax_out[i * self.k: (i + 1) * self.k] = \
                np.array([b == x for _, b in sorted(self.bins.items(), key=lambda t: t[0])])

        if sparse:
            ax_out = csc_matrix(ax_out)

        return ax_out

    def encode_2d_array(self, ax, sparse=False, orient='lines'):
        if orient == 'columns':
            ax = ax.transpose()

        ax_out = np.zeros([len(ax), len(ax[0]) * self.k])

        for i, ax_ in enumerate(ax):
            ax_out[i, :] = self.encode_1d_array(ax_)

        if orient == 'columns':
            ax_out = ax_out.transpose()

        if sparse:
            ax_out = csc_matrix(ax_out)

        return ax_out

    def decode_1d_array(self, ax, sparse=False):

        if sparse:
            ax = ax.toarray()[0]

        ax_out = np.zeros(int(len(ax) / self.k))

        for i in range(len(ax_out)):
            ax_out[i] = self.bins.get(np.where(ax[i * self.k: (i + 1) * self.k])[0][0])

        return ax_out

    def decode_2d_array(self, ax, sparse=False, orient='lines'):

        if orient == 'columns':
            ax = ax.transpose()

        if sparse:
            ax = ax.toarray()

        ax_out = np.zeros([len(ax), int(len(ax[0]) / self.k)])

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
            var = np.var(s)

            if var <= 0:
                raise ValueError('semi negative variance computed from signal passed')

            self.params['sigma'] = np.sqrt(var)

        elif self.method == 'max':
            max = np.max(s)

            if max < 0:
                max = - np.min(s)

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
