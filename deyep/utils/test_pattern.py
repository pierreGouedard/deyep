# Global imports
import numpy as np
import networkx as nx

# local import
from deyep.core.firing_graph.utils import mat_from_tuples
from deyep.core.firing_graph.graph import FiringGraph


class TestPattern(object):
    l_seq_types = ['det', 'rand']

    def __init__(self, name, input_vertices, core_vertices, output_vertices, depth):

        self.name = name
        self.input_vertices = input_vertices
        self.core_vertices = core_vertices
        self.output_vertices = output_vertices
        self.depth = depth

    def build_graph_pattern_final(self):
        """
        Return a firing graph as it should be at the end of the test

        :return:
        """
        raise NotImplementedError

    def build_graph_pattern_init(self):
        """
        Return a firing graph as it should be at the beginning of the test
        :return:
        """
        raise NotImplementedError

    def build_deterministic_io(self):
        """
        Build deterministic activation of input output pair as couple of numpy.array
        :return:
        """
        raise NotImplementedError

    def build_random_io(self):
        """
        Build random activation of input with all zeros output pair as couple of numpy.array
        :return:
        """
        raise NotImplementedError

    def init_io_sequence(self,):

        raise NotImplementedError

    def generate_io_sequence(self, length, p=0.5, seed=1830):
        """
        Return a mix between deterministic io and noisy io
        :return:
        """
        np.random.seed(seed)
        ax_isequence, ax_osequence = self.init_io_sequence()

        for _ in range(length):
            seqtype = np.random.choice(self.l_seq_types, p=[p, 1-p])

            if seqtype == 'det':
                # Add deterministic pattern
                ax_inputs, ax_outputs = self.build_deterministic_io()
                ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

            else:
                # Add random pattern
                ax_inputs, ax_outputs = self.build_random_io()
                ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

        return ax_isequence, ax_osequence


class AndPattern2(TestPattern):
    """
    The primary test purpose of this pattern is to test for edge removal in a firing graph of depth 2. After a
    significant number of iteration, only correct edges should remain
    """

    def __init__(self, ni, no, w=100, p=0.5, random_target=False, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Core params of test
        self.ni, self.no, self.w, self.p, self.random_target = ni, no, w, p, random_target,

        # Init
        self.firing_graph = None

        # Set targets
        self.target = [
            np.random.choice(range(self.ni * j, self.ni * (j + 1)), np.random.randint(1, int(0.5 * self.ni)), replace=False)
            for j in range(self.no)
        ]

        # Build list of vertices
        input_vertices = [['input_{}'.format((self.ni * i) + j) for j in range(self.ni)] for i in range(self.no)]
        core_vertices = ['core_{}'.format(i) for i in range(self.no)]
        output_vertices = ['output_{}'.format(i) for i in range(self.no)]
        TestPattern.__init__(self, 'and', input_vertices, core_vertices, output_vertices, 2)

    def build_graph_pattern_final(self):
        # Build edges
        l_edges = []
        for i in range(self.no):
            l_edges += [(self.input_vertices[i][j], self.core_vertices[i]) for j in range(self.ni)
                        if j + self.ni * i in self.target[i]]
        l_edges += zip(self.core_vertices, self.output_vertices)

        # Build Firing graph
        sax_I, sax_D, sax_O = mat_from_tuples(l_edges, self.ni * self.no, self.no, self.no, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices(
            'AndPatFinal', sax_D, sax_I, sax_O, 3, self.w, np.array([len(self.target[i]) for i in range(self.no)])
        )

        return self.firing_graph

    def build_graph_pattern_init(self):

        # Build edges
        l_edges = []
        for i in range(self.no):
            l_edges += [(self.input_vertices[i][j], self.core_vertices[i]) for j in range(self.ni)]
        l_edges += zip(self.core_vertices, self.output_vertices)

        # Build Firing graph
        sax_I, sax_D, sax_O = mat_from_tuples(l_edges, self.ni * self.no, self.no, self.no, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices('AndPat2', sax_D, sax_I, sax_O, self.depth + 1, self.w, [1, 1])

        return self.firing_graph

    def build_deterministic_io(self):
        ax_inputs = np.zeros((1, self.ni * self.no))

        for i in range(self.no):
            ax_inputs_ = np.zeros((1, self.ni * self.no))
            ax_inputs_[0, self.target[i]] = 1

            ax_inputs = np.vstack((ax_inputs, ax_inputs_))

        ax_outputs = np.eye(self.no)

        return ax_inputs[1:, :], ax_outputs

    def build_random_io(self):

        # Generate random inputs
        ax_inputs = np.random.binomial(1, self.p, (1, self.ni * self.no))

        if not self.random_target:
            for i in range(self.no):
                ax_inputs[0, self.target[i]] = 0

        # Generate outputs
        ax_outputs = np.zeros((1, self.no))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.ni * self.no)), np.zeros((1, self.no))

    def layout(self):
        pos = dict()

        pos.update({'inputs': {'pos': {i: (0, i) for i in range(self.ni * self.no)}, 'color': 'r'}})
        n = self.ni * self.no
        pos.update({'cores': {
            'pos': {(n + i): (1, ((self.ni * i) + (self.ni * (i + 1))) / 2) for i in range(self.no)},
            'color': 'k'}
        })
        n = self.ni * self.no + self.no
        pos.update({'outputs': {
            'pos': {(n + i): (2, ((self.ni * i) + (self.ni * (i + 1))) / 2) for i in range(self.no)},
            'color': 'b'}
        })

        return pos


class AndPattern3(TestPattern):
    """
    The primary test purpose of this pattern is to test for edge removal in a firing graph of depth 3. After a
    significant number of iteration, only correct edges should remain
    """
    def __init__(self, ni, no, w=100, p=0.5, n_selected=2, random_target=False, seed=None):

        try:
            assert n_selected < ni
        except AssertionError:
            raise ValueError('Number of input must be larger than the number of selected bits')

        if seed is not None:
            np.random.seed(seed)

        # Core params of test
        self.ni, self.no, self.nc, self.w, self.p, self.random_target = ni, no, 3, w, p, random_target

        # Init
        self.firing_graph = None

        # Set targets
        self.target = [
            np.random.choice(
                range(self.ni * j, self.ni * (j + 1)), np.random.randint(n_selected + 1, self.ni / 2), replace=False)
            for j in range(self.no)
        ]
        self.target_selected = [self.target[j][:n_selected] for j in range(self.no)]

        # Build list of vertices
        input_vertices = [['input_{}'.format((self.ni * i) + j) for j in range(self.ni)] for i in range(self.no)]
        core_vertices = [
            ['core_{}'.format(self.nc * i), 'core_{}'.format(self.nc * i + 1), 'core_{}'.format(self.nc * i + 2)]
            for i in range(self.no)
        ]
        output_vertices = ['output_{}'.format(i) for i in range(self.no)]
        TestPattern.__init__(self, 'and', input_vertices, core_vertices, output_vertices, 3)

    def build_graph_pattern_final(self):
        # Build edges
        l_edges, ax_levels = [], np.ones(self.nc * self.no)

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][0]) for j in range(self.ni)
                if self.ni * i + j in self.target_selected[i]
            ]
            ax_levels[i * self.nc] = len(self.target_selected[i])

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][1]) for j in range(self.ni)
                if self.ni * i + j not in self.target_selected[i] and self.ni * i + j in self.target[i]
            ]

        for i in range(self.no):
            l_edges += [
                (self.core_vertices[i][0], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][1], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][self.nc - 1], self.output_vertices[i])
            ]
            ax_levels[i * self.nc + self.nc - 1] = self.nc - 1

        # Build Firing graph
        sax_I, sax_D, sax_O = mat_from_tuples(l_edges, self.ni * self.no, self.no * self.nc, self.no, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices('AndPat3', sax_D, sax_I, sax_O, self.depth + 1, self.w, ax_levels)

        return self.firing_graph

    def build_graph_pattern_init(self):

        # Build edges
        l_edges, ax_levels = [], np.ones(self.nc * self.no)

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][0]) for j in range(self.ni)
                if self.ni * i + j in self.target_selected[i]
            ]
            ax_levels[i * self.nc] = len(self.target_selected[i])

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][1]) for j in range(self.ni)
                if self.ni * i + j not in self.target_selected[i]
            ]

        for i in range(self.no):
            l_edges += [
                (self.core_vertices[i][0], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][1], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][self.nc - 1], self.output_vertices[i])
            ]
            ax_levels[i * self.nc + self.nc - 1] = self.nc - 1

        # Build Firing graph
        sax_I, sax_D, sax_O = mat_from_tuples(l_edges, self.ni * self.no, self.no * self.nc, self.no, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices('AndPatInit', sax_D, sax_I, sax_O, self.depth + 1, self.w, ax_levels)

        return self.firing_graph

    def build_deterministic_io(self):
        ax_inputs = np.zeros((1, self.ni * self.no))

        for i in range(self.no):
            ax_inputs_ = np.zeros((1, self.ni * self.no))
            ax_inputs_[0, self.target[i]] = 1

            ax_inputs = np.vstack((ax_inputs, ax_inputs_))

        ax_outputs = np.eye(self.no)

        return ax_inputs[1:, :], ax_outputs

    def build_random_io(self):

        # Generate random inputs
        ax_inputs = np.random.binomial(1, self.p, (1, self.ni * self.no))

        if not self.random_target:
            for i in range(self.no):
                ax_inputs[0, self.target[i]] = 0

        # Generate outputs
        ax_outputs = np.zeros((1, self.no))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.ni * self.no)), np.zeros((1, self.no))

    def layout(self):
        pos = dict()

        pos.update({'inputs': {'pos': {i: (0, i) for i in range(self.ni * self.no)}, 'color': 'r'}})
        n = self.ni * self.no

        for i in range(self.no):
            pos.update({'cores': {
                'pos': {(n + (i * self.nc)): (1, ((self.ni * i) + (self.ni * (i + 1))) / 4),
                        (n + + (i * self.nc) + 1):  (1, (((self.ni * i) + (self.ni * (i + 1))) * 3) / 4),
                        (n + + (i * self.nc) + 2): (2, ((self.ni * i) + (self.ni * (i + 1))) / 2)},
                'color': 'k'}
            })
            n += self.nc

        pos.update({'outputs': {
            'pos': {(n + i): (3, ((self.ni * i) + (self.ni * (i + 1))) / 2) for i in range(self.no)},
            'color': 'b'}
        })

        return pos


class RandomPattern(TestPattern):
    """
    random generation with the correct convergence sentence
    """
    def __init__(self, ni, nd, no, depth, p, seed=666):

        self.ni, self.nd, self.no, self.depth, self.p, self.seed = ni, nd, no, depth, p, seed

        # Build list of vertices
        input_vertices = ['input_{}'.format(i) for i in range(self.ni)]
        core_vertices = ['core_{}'.format(i) for i in range(self.nd)]
        output_vertices = ['output_{}'.format(i) for i in range(self.no)]

        TestPattern.__init__(self, 'random', input_vertices, core_vertices, output_vertices, self.depth)

    def build_graph_pattern_final(self):
        raise NotImplementedError

    def build_graph_pattern_init(self, decrease='linear'):
        np.random.seed(self.seed)

        l_selected = [c for c in self.core_vertices if np.random.choice([0, 1], p=[1 - self.p, self.p]) == 1]
        l_candidates = [n for n in self.core_vertices if n not in l_selected]
        l_edges = [(np.random.choice(self.input_vertices), c) for c in l_selected]

        for i in range(100):

            if decrease == 'linear':
                p = self.p / (i + 2)

            elif decrease == 'exponential':
                p = pow(self.p, i)

            l_selected_ = [c for c in l_candidates if np.random.choice([0, 1], p=[1 - p, p]) == 1]

            if len(l_selected_) == 0:
                break

            l_edges += [(np.random.choice(l_selected), c) for c in l_selected_]
            l_selected = list(l_selected_)
            l_candidates = [n for n in self.core_vertices if n not in l_selected]

        return l_edges

    def build_deterministic_io(self):
        return self.build_random_io()

    def build_random_io(self):
        ax_inputs = np.array([np.random.choice([0, 1]) for _ in range(self.ni)])
        ax_outputs = np.array([np.random.choice([0, 1]) for _ in range(self.ni)])

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.ni)), np.zeros((1, self.no))

    def layout(self, ax_graph):
        pos, pos_ = {}, nx.spring_layout(nx.from_numpy_array(ax_graph))
        pos.update({'inputs': {'pos': {i: pos_[i] for i in range(self.ni)}, 'color': 'r'}})
        pos.update({'networks': {'pos': {self.ni + i: pos_[self.ni + i] for i in range(self.nd)}, 'color': 'k'}})
        pos.update({'outputs': {'pos': {self.ni + self.nd + i: pos_[self.ni + self.nd + i] for i in range(self.no)},
                                'color': 'b'}})

        return pos



