# Global imports
import numpy as np
import unittest

# Local import
from deyep.utils.signal_processing.various import Discretizer, Normalizer

__maintainer__ = 'Pierre Gouedard'


class TestVariousUtils(unittest.TestCase):
    # TODO: refactor
    def setUp(self):

        # create random signal
        self.s = np.random.randn(1000) + 1e3

        # Create random matrix
        self.m = np.random.randn(100, 100)

        # Set discretization params
        self.N = 100

        # Create normalizer and discretizer
        self.normalizer = Normalizer()
        self.discretizer = Discretizer(self.N)

    def test_normalizer(self):
        """

        python -m unittest tests.signal_processing.various.TestVariousUtils.test_normalizer

        """

        # Test variance normalization
        self.normalizer.method = 'variance'
        s_ = self.normalizer.set_transform(self.s)

        assert np.abs(np.std(s_) - 1) < 1e-1

        # Test max normalization
        self.normalizer.method = 'max'
        s_ = self.normalizer.set_transform(self.s)
        _s = self.normalizer.set_transform(- self.s)

        assert np.abs(np.max(s_) - 1) < 1e-10
        assert np.abs(np.min(_s) + 1) < 1e-10

        # Test sum normalization
        self.normalizer.method = 'sum'
        s_ = self.normalizer.set_transform(self.s)

        assert np.abs(np.sum(s_) - 1) < 1e-10

    def test_dicretizer_1d_dense(self):
        """

        python -m unittest tests.signal_processing.various.TestVariousUtils.test_dicretizer

        """
        # Center signal
        self.s -= 1e3

        # Test dicretize 1d array with treshold
        self.discretizer.set_discretizer_bins(s=self.s, method='treshold', **{'treshold': 1e-1})

        assert len(self.discretizer.bins) == self.N

        s_ = self.discretizer.discretize_array(self.s)

        assert all([x in self.discretizer.bins.values() for x in np.unique(s_)])
        assert 0 in np.unique(s_)

        s_ = self.discretizer.encode_1d_array(self.s)

        assert len(s_) == self.N * len(self.s)
        assert all([sum(s_[i * self.N: (i + 1) * self.N]) == 1 for i in range(len(self.s))])

        # Test perfect recovery of discretized values on decoding
        _s = self.discretizer.decode_1d_array(s_)
        s_ = self.discretizer.discretize_array(self.s)

        assert (s_ == _s).all()

    def test_dicretizer_1d_sparse(self):
        """

        python -m unittest tests.signal_processing.various.TestVariousUtils.test_dicretizer

        """
        # Center signal
        self.s -= 1e3

        # Test dicretize 1d array with treshold
        self.discretizer.set_discretizer_bins(s=self.s, method='treshold', **{'treshold': 1e-1})

        # Test with sparse settings
        s_ = self.discretizer.encode_1d_array(self.s, sparse=True)

        assert s_.shape[0] * s_.shape[1] == self.N * len(self.s)
        assert all([s_[0, i * self.N: (i + 1) * self.N].sum() == 1 for i in range(len(self.s))])

        # Test perfect recovery on decoding
        _s = self.discretizer.decode_1d_array(s_, sparse=True)
        s_ = self.discretizer.discretize_array(self.s)

        assert (s_ == _s).all()

    def test_dicretizer_2d_dense(self):
        """

        python -m unittest tests.signal_processing.various.TestVariousUtils.test_dicretizer

        """

        # Test discretize 2d arrays
        self.discretizer.set_discretizer_bins(s=self.m.flatten('F'), method='treshold', **{'treshold': 1e-1})

        m_ = self.discretizer.encode_2d_array(self.m, orient='columns')

        assert len(m_[:, 0]) == self.N * len(self.m[:, 0])
        for i in range(len(self.m[0])):
            assert all([sum(m_[i * self.N: (i + 1) * self.N, i]) == 1 for i in range(len(self.m))])

        # Test perfect recovery of discretized values on decoding
        _m = self.discretizer.decode_2d_array(m_, orient='columns')
        m_ = self.discretizer.discretize_array(self.m)

        assert (m_ == _m).all()

    def test_dicretizer_2d_sparse(self):
        """

        python -m unittest tests.signal_processing.various.TestVariousUtils.test_dicretizer

        """

        # Test discretize 2d arrays
        self.discretizer.set_discretizer_bins(s=self.m.flatten('F'), method='treshold', **{'treshold': 1e-1})

        # Test in sparse settings
        m_ = self.discretizer.encode_2d_array(self.m, sparse=True, orient='columns')

        assert m_[:, 0].shape[0] * m_[:, 0].shape[1] == self.N * len(self.m[:, 0])
        for i in range(len(self.m[0])):
            assert all([m_[i * self.N: (i + 1) * self.N, i].sum() == 1 for i in range(len(self.m))])

        # Test perfect recovery of discretized values on decoding
        _m = self.discretizer.decode_2d_array(m_, sparse=True, orient='columns')
        m_ = self.discretizer.discretize_array(self.m)

        assert (m_ == _m).all()




