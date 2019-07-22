# Global import
import numpy as np
import random

# local import
from deyep.core.firing_graph.utils import mat_from_tuples
from deyep.core.firing_graph.graph import FiringGraph


class Sampler(object):
    # Firing Graph of depth 2
    capacity_init = 3

    # Firing Graph of depth 3
    capacity_core = 5

    def __init__(self, size, w, imputer, selected_bits=None, preselected_bits=None, supervised=True, verbose=0):
        """

        :param size: list [#input, #output]
        :param w: int weights of edges of firing graph
        :param imputer: deyep.core.imputer.comon.Imputer
        :param selected_bits: dict of set of inputs index already sampled in previous iteration (key = output index)
        :param preselected_bits: dict of set of inputs index from which we want to draw next sample (key = output index)
        :param supervised: bool
        :param verbose: int
        """
        # Core params
        self.ni, self.no = size[0], size[1]
        self.w = w
        self.firing_graph = None
        self.verbose = verbose
        self.supervised = supervised

        # Get list of preselected and already selected bits if any
        if preselected_bits is None:
            self.preselect_bits = {}
        else:
            self.preselect_bits = preselected_bits

        if selected_bits is None:
            self.selected_bits = {}
        else:
            self.selected_bits = selected_bits

        # utils
        self.imputer = imputer

    def reset_imputer(self):
        self.imputer.stream_features()
        return self

    def sample(self):

        if self.preselect_bits is None:
            if self.supervised:
                self.sample_supervised()
            else:
                raise NotImplementedError

        return self

    def sample_supervised(self):

        ax_selected = np.zeros(self.no, dtype=bool)

        # Select bits for each output
        while ax_selected.sum() != len(ax_selected):
            sax_si = self.imputer.stream_next_forward()
            sax_got = self.imputer.stream_next_backward()

            for i in sax_got.nonzero()[-1]:
                if not ax_selected[i]:
                    self.preselect_bits[i] = set(sax_si.nonzero()[1])

                    # Remove already selected bits
                    if self.selected_bits:
                        self.preselect_bits[i] = self.preselect_bits[i].difference(set(self.selected_bits[i]))

                    ax_selected[i] = True

        return self

    def build_graph_multiple_output(self, name='sampler'):
        """

        :return:
        """
        # Init parameter of the firing graph
        l_edges, l_weights, n_core, d_levels = [], [], 0, {}

        if self.selected_bits is not None:
            capacity = Sampler.capacity_core
        else:
            capacity = Sampler.capacity_init

        # Build matrices of the firing graph
        for i in range(self.no):
            l_edges, l_weights, n_core, d_levels = self.build_graph(i, l_edges, l_weights, n_core, d_levels)

        sax_I, sax_D, sax_O = mat_from_tuples(l_edges, self.ni, n_core, self.no, weights=l_weights)

        # Build level array
        ax_levels = np.zeros(n_core)
        for i, v in d_levels.items():
            ax_levels[i] = v

        # Build firing graph
        self.firing_graph = FiringGraph.from_matrices(name, sax_D, sax_I, sax_O, capacity, self.w, ax_levels)

        return self

    def build_graph(self, i, l_edges, l_weights, n_core, d_levels):

        # Create first layer of graph (input vertices)
        for pb in self.preselect_bits[i]:
            l_edges += [('input_{}'.format(pb), 'core_{}'.format(n_core))]
            l_weights += [self.w]

        # If some bits have already been selected
        if self.selected_bits:
            for b in self.selected_bits[i]:
                l_edges += [('input_{}'.format(b), 'core_{}'.format(n_core + 1))]
                l_weights += [np.float('inf')]

            # Add core edges
            l_edges += [
                ('core_{}'.format(n_core), 'core_{}'.format(n_core + 2)),
                ('core_{}'.format(n_core + 1), 'core_{}'.format(n_core + 2)),
                ('core_{}'.format(n_core + 2), 'output_{}'.format(i))
            ]
            l_weights += [np.float('inf')] * 3

            # Update levels
            d_levels.update({n_core: 1, n_core + 1: len(self.selected_bits[i]), n_core + 2: 2})
            n_core += 3

        else:
            # Add core edges and update levels
            l_edges += [('core_{}'.format(n_core), 'output_{}'.format(i))]
            l_weights += [np.float('inf')]
            d_levels.update({n_core: 1})

            n_core += 1

        return l_edges, l_weights, n_core, d_levels