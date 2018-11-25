# Global imports
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from deyep.core.builder.comon import mat_from_tuples
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.imputer.identity import DoubleIdentityImputer
from deyep.core.solver.canonical import CanonicalDeepNetSolver
from deyep.utils.driver.nmp import NumpyDriver
from tests.utils import testPattern as tp

__maintainer__ = 'Pierre Gouedard'


class TestSolverConvergence(unittest.TestCase):
    def setUp(self):

        # Set core parameters
        self.l0, self.tau, self.wo = 1000, 5, 10

        # Create random pattern with canonical basis
        self.rand_pat = tp.RandomPattern(10, 100, 10, 2, 0.6)

        mat_in, mat_net, mat_out = mat_from_tuples(self.rand_pat.build_graph_pattern_init(), self.rand_pat.ni,
                                                   self.rand_pat.nd, self.rand_pat.no,
                                                   weights=[10] * len(self.rand_pat.build_graph_pattern_init()))

        self.rand_dn_c = DeepNetwork.from_matrices('test', mat_net, mat_in, mat_out, 10, 'canonical', w0=self.wo, l0=self.l0,
                                                   tau=self.tau)

    def test_random_pattern_low_penalty(self):
        """
        python -m unittest tests.core.solver.convergence.TestSolverConvergence.test_random_pattern_low_penalty

        """
        # Create temporary directory for test
        driver = NumpyDriver()

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.rand_pat.generate_io_sequence(1000, seed=1234)
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat')

        # Create solver and run epoch
        solver = CanonicalDeepNetSolver(self.rand_dn_c, self.rand_pat.delay, imputer, p0=1.,  verbose=1)
        solver.fit_epoch(500)

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        Ow = solver.deep_network.Ow
        ax_values = np.array([Ow[i, j] for i, j in zip(*Ow.nonzero()) if Ow[i, j] < 20])

        # INFO the mean is likely to be on the left of theoretical mean: due to level of nodes (last signal received
        # after deactivation of level most be a negative, thus, introduce left biase)
        self.assertAlmostEqual(self.wo, ax_values.mean(), delta=0.5)

        # Now test fitted network
        ax_input_, ax_output_ = np.ones([100, len(self.rand_pat.input_nodes)]), np.zeros(1)
        imputer_, tmpdirin, tmpdirout = create_imputer(ax_input_, ax_output_, driver, name='rand_pat')

        # Transform inputs by fitted network
        sax_output = csc_matrix(solver.transform(imputer_, n=100))

        # Assert that no output has been triggered
        self.assertTrue(sax_output[10:, :].toarray().all().all())

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

    def test_random_pattern_high_penalty(self):
        """
        python -m unittest tests.core.solver.convergence.TestSolverConvergence.test_random_pattern_high_penalty

        """
        # Create temporary directory for test
        driver = NumpyDriver()

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.rand_pat.generate_io_sequence(1000, seed=1234)
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat')

        # Create solver and run epoch
        solver = CanonicalDeepNetSolver(self.rand_dn_c, self.rand_pat.delay, imputer, p0=10)
        solver.fit_epoch(500)

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        # Now test fitted network
        ax_input, ax_output = np.zeros([100, len(self.rand_pat.input_nodes)]), np.zeros(1)
        ax_input[0, :] = 1
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat')

        # Transform inputs by fitted network
        ax_output = solver.transform(imputer, n=100)

        # Assert that no output has been triggered
        self.assertTrue(not ax_output.any().any())

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()


def create_imputer(ax_input, ax_output, driver, name='test'):

    tmpdirin, tmpdirout = driver.TempDir(name, suffix='in', create=True), \
                          driver.TempDir(name, suffix='out', create=True)

    # Create I/O and save it into tmpdir files
    driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
    driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

    # Create and init imputer
    imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')
    imputer.stream_features()

    return imputer, tmpdirin, tmpdirout
