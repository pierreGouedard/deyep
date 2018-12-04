# Global imports
import unittest
from scipy.sparse import csc_matrix
from random import choice
import numpy as np

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.core.solver.filler import DeepNetFiller
from deyep.core.solver.drainer import DeepNetDrainer
from deyep.core.runner.comon import DeepNetRunner
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.utils.driver.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestFiller(unittest.TestCase):
    def setUp(self):

        # Set core params of deep network
        self.l0, self.tau, self.w0, self.capacity = 10, 5, 10, 5

        # Build synthetic graph
        sax_I = csc_matrix([[True, False], [False, False], [False, False]])
        sax_D = csc_matrix([[False, True], [False, False]])
        sax_0 = csc_matrix([[False, False, False], [True, True, True]])

        # Create deep network
        self.deep_network = DeepNetwork.from_matrices(
            'test_filler', sax_D, sax_I, sax_0, 5, w0=self.w0, l0=self.l0, tau=self.tau
        )

        # Create imputer
        self.sax_in = csc_matrix([
            [True, False, False], [False, True, True], [True, False, True], [True, True, True], [False, True, False],
            [False, False, True], [True, True, False], [True, False, False], [True, False, False]
        ])

        self.sax_out = csc_matrix([[True, True, False], [False, False, True], [False, False, True], [False, False, True],
                                   [False, False, True], [False, False, True], [False, False, True], [True, True, False],
                                   [True, True, False]])

        np.random.seed(1234)

    def activity_tracking(self):
        """
        python -m unittest tests.core.filler.TestFiller.activity_tracking

        """

        # Create imputer
        imputer = create_imputer(self.sax_in, self.sax_out)

        # Create filler
        filler = DeepNetFiller(self.deep_network, imputer, {'depth': 3})

        # Check initialisation
        self.assertEqual(filler.d_outputs, {1: [0, 1, 2]})

        # Record activity of nodes
        sax_sn, _, _ = record_activity(filler)

        # Check tracking of activity
        self.assertTrue(
            (sax_sn == csc_matrix([[True], [False], [True], [True], [False], [False], [True], [True], [True]]))
            .toarray()
            .all()
        )

    def explode_level(self):
        """
        python -m unittest tests.core.filler.TestFiller.explode_level

        """

        sax_l1 = csc_matrix([[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1]]).transpose()
        sax_l2 = csc_matrix([[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]]).transpose()
        sax_out = DeepNetFiller.explode_level(sax_l1 + sax_l2)

        self.assertTrue((sax_out[:, 0] == (sax_l1 - sax_l2)).toarray().all())
        self.assertTrue((sax_out[:, 1] == sax_l2).toarray().all())

    def level_building(self):
        """
        python -m unittest tests.core.filler.TestFiller.level_building

        """
        # Initialisation
        filler = DeepNetFiller(self.deep_network, create_imputer(self.sax_in, self.sax_out), {'depth': 3})
        sax_sn, d_map, sax_so = record_activity(filler)

        # Case 1
        d_level = {1: True, 2: True}
        sax_sn = enrich_signal_levels(sax_sn, d_level.keys())

        _, d_level_1 = filler.build_output(sax_sn, sax_so.astype(int), d_level)
        _, d_level_2 = filler.build_output(sax_sn, sax_so.astype(int), d_level, init=False)

        self.assertTrue(d_level_1[1] and not d_level_1[2])
        self.assertTrue(all([d_level_1[i] for i in range(3, max(d_level_1.keys()))]))
        self.assertTrue(not d_level_2[1] and d_level_2[2])
        self.assertTrue(not any([d_level_2[i] for i in range(3, max(d_level_1.keys()))]))

        # Case 2
        d_level = {2: True, 4: True}
        sax_sn = enrich_signal_levels(sax_sn, d_level.keys())

        _, d_level_1 = filler.build_output(sax_sn, sax_so.astype(int), d_level)
        _, d_level_2 = filler.build_output(sax_sn, sax_so.astype(int), d_level, init=False)

        self.assertTrue(d_level_1[2] and d_level_1[4])
        self.assertTrue(not any([d_level_1[i] for i in range(5, max(d_level_1.keys()))]))
        self.assertTrue(d_level_2[2] and not d_level_2[3] and not d_level_2[4])
        self.assertTrue(all([d_level_2[i] for i in range(5, max(d_level_1.keys()))]))

        # Case 3
        d_level = {1: True, 2: True, 4: True, 5: True}
        sax_sn = enrich_signal_levels(sax_sn, d_level.keys())

        _, d_level_1 = filler.build_output(sax_sn, sax_so.astype(int), d_level)
        _, d_level_2 = filler.build_output(sax_sn, sax_so.astype(int), d_level, init=False)

        self.assertTrue(d_level_1[1] and d_level_1[2] and not d_level_1[3] and not d_level_1[4] and d_level_1[5])
        self.assertTrue(not any([d_level_1[i] for i in range(6, max(d_level_1.keys()))]))
        self.assertTrue(not d_level_2[1] and d_level_2[2] and not d_level_2[3] and d_level_2[4] and not d_level_2[5])
        self.assertTrue(all([d_level_2[i] for i in range(6, max(d_level_1.keys()))]))

    def output_building(self):
        """
        python -m unittest tests.core.filler.TestFiller.output_building

        """

        # Initialisation
        filler = DeepNetFiller(self.deep_network, create_imputer(self.sax_in, self.sax_out), {'depth': 3})
        sax_sn, d_map, sax_so = record_activity(filler)

        # Test output construciton
        sax_out_sub, d_level_out, _ = build_output(filler, sax_sn, sax_so, d_map)

        # Check correctness of construction
        self.assertTrue(
            (sax_out_sub == csc_matrix([[False], [False], [True], [True], [False], [False], [True], [False], [False]]))
            .toarray()
            .all()
        )
        self.assertTrue(d_level_out[1])
        self.assertTrue(not any([d_level_out[i] for i in range(2, max(d_level_out.keys()))]))

        sax_out_sub, d_level_out, _ = build_output(filler, sax_sn, sax_so, d_map, init=False)

        # Check correctness of construction
        self.assertTrue(
            (sax_out_sub == csc_matrix([[True], [False], [False], [False], [False], [False], [False], [True], [True]]))
            .toarray()
            .all()
        )
        self.assertTrue(not d_level_out[1])
        self.assertTrue(all([d_level_out[i] for i in range(2, max(d_level_out.keys()))]))

    def graph_building(self):
        """
        python -m unittest tests.core.filler.TestFiller.graph_building

        """

        # Create params  imputer
        imputer, tmpdirin, tmpdirout = create_imputer(self.sax_in, self.sax_out, return_dir=True)
        d_params = {
            'depth': 3, 'ni': self.deep_network.I.shape[0], 'nd': 10, 'p0': 0.1, 'w0': self.w0, 'l0': self.l0,
            'tau': self.tau, 'capacity': self.capacity
        }

        filler = DeepNetFiller(self.deep_network, imputer, d_params)\
            .build_imputer_fill()\
            .build_graph_fill(BinomialGraphBuilder, 1234)

        # Drain network
        filler.imputer_fill.stream_features(is_cyclic=True)
        drainer = DeepNetDrainer(filler.deep_network_fill, 2, filler.imputer_fill, p0=1, verbose=0)
        drainer.fit_epoch(100)

        # Clean tmp dir
        tmpdirin.remove(), tmpdirout.remove(), filler.remove_tmpdir()

        # Assert that no link creation respect rules
        nio, _ = len(filler.d_mapping.pop('ido')), len(filler.d_mapping.pop('ide'))
        l_io, l_ie = [x[0]['id'] for x in filler.d_mapping.values()], \
            [x[1]['id'] for x in filler.d_mapping.values() if len(x) > 1]

        self.assertEqual(filler.deep_network_fill.O[:nio, l_ie].sum(), 0)
        self.assertEqual(filler.deep_network_fill.O[nio:, l_io].sum(), 0)


def record_activity(filler):

    runner = DeepNetRunner(
        filler.deep_network_original, filler.params_network['depth'], filler.imputer_original
    )
    sax_sn, d_map = runner.track_activity(filler.params_network['depth'] - 1, l_nodes=filler.d_outputs.keys())
    sax_so = filler.imputer_original.features_backward

    return sax_sn, d_map, sax_so


def enrich_signal_levels(sax_sn, l_levels):
    sax_sn_ = csc_matrix((sax_sn.shape[0], max(l_levels)))

    for i, _ in zip(*sax_sn.nonzero()):
        j = choice(l_levels)
        sax_sn_[i, j - 1] = 1

    return sax_sn_


def build_output(filler, sax_sn, sax_so, d_map, init=True):

    id_, l_outputs = filler.d_outputs.items()[0]
    sax_sn_sub = filler.explode_level(sax_sn[:, d_map[id_]])
    sax_so_sub = sax_so[:, l_outputs]

    # Get level information
    d_level = filler.deep_network_original.network_nodes[id_].d_levels
    l_max = filler.deep_network_original.D[:, id_].astype(int).sum()
    d_level_sub = {k: v for k, v in d_level.items() if (d_level[k] and k < l_max) or k == l_max}

    # Update output
    sax_out_sub, d_level_out = filler.build_output(sax_sn_sub, sax_so_sub.astype(int), d_level_sub, init=init)

    return sax_out_sub, d_level_out, filler


def create_imputer(sax_in, sax_out, return_dir=False):

    # Create temporary directory for test
    driver = NumpyDriver()
    tmpdirin, tmpdirout = driver.TempDir('test_filler', suffix='in', create=True), \
        driver.TempDir('test_filler', suffix='out', create=True)

    # Create I/O and save it into tmpdir files
    driver.write_file(sax_in, driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
    driver.write_file(sax_out, driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

    # Create and init imputer
    imputer = DoubleArrayImputer('test_filler', tmpdirin.path, tmpdirout.path)
    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')
    imputer.stream_features(is_cyclic=False)

    if return_dir:
        return imputer, tmpdirin, tmpdirout

    tmpdirin.remove()
    tmpdirout.remove()

    return imputer