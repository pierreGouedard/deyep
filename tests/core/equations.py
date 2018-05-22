# Global imports
import unittest

# Local import
from tests.comon import get_mat_from_path
from deyep.core.constructors.constructors import Constructor
from deyep.core.tools.linear_algebra import inner_product

__maintainer__ = 'Pierre Gouedard'


class TestEquations(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        n_i, n_rn, n_o, self.capacity = 2, 5, 3, 2
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices from list of edges and build network
        self.mat_in, self.mat_net, self.mat_out = get_mat_from_path(l_edges, n_i, n_rn, n_o)
        self.deep_network_1 = Constructor.from_weighted_direct_matrices(mat_net, mat_in, mat_out, self.capacity)

    def test_forward_transmission(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_transmission

        """

        # Test FNT

        # Test FOT

        raise NotImplementedError

    def test_forward_processing(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_processing

        """
        # Test FNP

        # Test FLP

        # Test FCP

        raise NotImplementedError

    def test_backward_transmission(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_backward_transmission

        """

        # Test BNT

        # Test BIT

        raise NotImplementedError

    def test_backward_processing(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_backward_processing

        """

        # Test BOP

        # Test BCP

        # Test BNP

        # Test BLP

        raise NotImplementedError

    def test_network_update(self):
        """
         python -m unittest tests.core.equations.TestEquations.test_network_update

         """

        # Test BDU

        # Test BOU

        # Test BIU

        raise NotImplementedError






