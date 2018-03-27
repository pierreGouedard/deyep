# Global import
import numpy as np


class Discretizer:

    def __init__(self, k, method='bins', bins=None):
        self.method = method
        self.k = k
        self.bins = bins

    def set_bins_from_signal(self, s):

        max_, min_ = max(s), min(s)
        self.set_bins_from_bound(max_, min_)

    def set_bins_from_bound(self, max_, min_):

        self.bins = {i: x for i, x in enumerate(np.linspace(min_, max_, self.k))}

    def discretize_value(self, x):

        if self.bins is None:
            raise ValueError('First set the bins of discretizer')

        x_ = min([(v, abs(x - v)) for k, v in  self.bins.items()], key=lambda t: t[1])[0]

        return x_

    def discretize_array(self, ax):

        # Vectorize discretie value function
        vdicretizer = np.vectorize(lambda x: self.discretize_value(x))

        # Apply to array
        ax_ = vdicretizer(ax)

        return ax_




