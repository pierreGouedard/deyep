# Global imports
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from deyep.core.builder.comon import mat_from_tuples
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.imputer.identity import DoubleIdentityImputer
from deyep.core.solver.comon import DeepNetSolver
from deyep.core.runner.comon import DeepNetRunner
from deyep.utils.driver.nmp import NumpyDriver
from tests.utils import testPattern as tp

__maintainer__ = 'Pierre Gouedard'


class TestSolverConvergence(unittest.TestCase):
    def setUp(self):

        # Set core parameters
        self.l0, self.tau, self.wo = 1000, 5, 10

        # Create random pattern with canonical basis
        self.rand_pat = tp.RandomPattern(10, 100, 10, 2, 0.6)

        mat_in, mat_net, mat_out = mat_from_tuples(
            self.rand_pat.build_graph_pattern_init(), self.rand_pat.ni, self.rand_pat.nd, self.rand_pat.no,
            weights=[10] * len(self.rand_pat.build_graph_pattern_init())
        )

        self.rand_dn_c = DeepNetwork.from_matrices(
            'test_convergence', mat_net, mat_in, mat_out, 10, w0=self.wo, l0=self.l0, tau=self.tau
        )

    def low_penalty(self):
        """
        validate asymptotic Expected behaviour of deep network for low penalty
        python -m unittest tests.core.convergence.TestSolverConvergence.low_penalty

        """
        # Create temporary directory for test
        driver = NumpyDriver()

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.rand_pat.generate_io_sequence(5000, seed=234)
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat')

        # Create solver and run epoch
        solver = DeepNetSolver(self.rand_dn_c, self.rand_pat.delay, imputer, p0=1., verbose=1)
        solver.fit_epoch(500)

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        Ow = solver.deep_network.Ow
        ax_values = np.array([Ow[i, j] for i, j in zip(*Ow.nonzero()) if Ow[i, j] < 20])

        # TODO: Test doesn't pass find out why
        self.assertAlmostEqual(self.wo, ax_values.mean(), delta=0.5)

        # Now test fitted network
        ax_input, ax_output = np.ones([100, len(self.rand_pat.input_nodes)]), np.zeros(1)
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat', is_cyclic=False)

        # Transform inputs by fitted network
        runner = DeepNetRunner(solver.deep_network.copy(), 2, imputer)
        sax_output = csc_matrix(runner.transform_array())

        # Assert that no output has been triggered
        self.assertTrue(sax_output[10:, :].toarray().all().all())

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

    def high_penalty(self):
        """
        Validate asymptotic Expected behaviour of deep network with high penalty
        python -m unittest tests.core.convergence.TestSolverConvergence.high_penalty

        """
        # Create temporary directory for test
        driver = NumpyDriver()

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.rand_pat.generate_io_sequence(1000, seed=1234)
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat')

        # Create solver and run epoch
        solver = DeepNetSolver(self.rand_dn_c, self.rand_pat.delay, imputer, p0=10, verbose=1)
        solver.fit_epoch(500)

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        # Now test fitted network
        ax_input, ax_output = np.zeros([100, len(self.rand_pat.input_nodes)]), np.zeros(1)
        ax_input[0, :] = 1
        imputer, tmpdirin, tmpdirout = create_imputer(ax_input, ax_output, driver, name='rand_pat', is_cyclic=False)

        # Transform inputs by fitted network
        runner = DeepNetRunner(solver.deep_network.copy(), 2, imputer)
        ax_output = runner.transform_array()

        # Assert that no output has been triggered
        self.assertTrue(not ax_output.any().any())

        # Remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()


def create_imputer(ax_input, ax_output, driver, name='test', is_cyclic=True):

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
    imputer.stream_features(is_cyclic=is_cyclic)

    return imputer, tmpdirin, tmpdirout
