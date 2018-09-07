

# Global imports
import numpy as np
import networkx as nx

# local import


class TestPattern(object):
    l_seq_types = ['det', 'rand']

    def __init__(self, name, input_nodes, network_nodes, output_nodes, delay):

        self.name = name
        self.input_nodes = input_nodes
        self.network_nodes = network_nodes
        self.output_nodes = output_nodes
        self.delay = delay

    def build_graph_pattern_final(self):
        raise NotImplementedError

    def build_graph_pattern_init(self):
        raise NotImplementedError

    def build_deterministic_io(self):
        raise NotImplementedError

    def build_random_io(self):
        raise NotImplementedError

    def init_io_sequence(self,):
        raise NotImplementedError

    def generate_io_sequence(self, length, p=0.5, seed=1830):
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


class TreePattern(TestPattern):
    """
    The primary test purpose of this pattern is to test for connection of network nodes with outputs nodes. It is also
    convenient to test the forward backward synchronisation. after a significant number of iteration, 1 to 1 edges from
    network and output nodes should appears
    """
    def __init__(self, d, k):

        self.d = d
        self.k = k
        self.d_nodes = {1: range(self.k)}
        self.d_nodes.update({d + 2: range(sum(pow(self.k, x) for x in range(1, d + 2)),
                                          sum(pow(self.k, x) for x in range(1, d + 3)))
                             for d in range(self.d - 1)})

        # Build list of nodess
        input_nodes = ['input_0']
        network_nodes = ['network_{}'.format(i) for i in range(self.get_size(d, k))]
        output_nodes = ['output_{}'.format(i) for i in range(self.get_size(d, k))]

        TestPattern.__init__(self, 'tree', input_nodes, network_nodes, output_nodes, 2)

    @staticmethod
    def get_size(d, k):
        return sum([pow(k, i + 1) for i in range(d)])

    def build_graph_pattern_final(self):

        # Build input edges
        l_edges = []
        for i in range(self.k):
            l_edges += [(self.input_nodes[0], self.network_nodes[i])]

        for k, v in sorted(self.d_nodes.items(), key=lambda (k_, v_): k_)[:-1]:
            for n, i in enumerate(sorted(v)):
                for j in range(self.k):
                    n_child = sorted(self.d_nodes[k + 1])[j + (n * self.k)]
                    l_edges += [(self.network_nodes[i], self.network_nodes[n_child])]

        # Build output edges
        l_edges += zip(self.network_nodes, self.output_nodes)

        return l_edges

    def build_graph_pattern_init(self):

        # Build input edges
        l_edges = []
        for i in range(self.k):
            l_edges += [(self.input_nodes[0], self.network_nodes[i])]

        for k, v in sorted(self.d_nodes.items(), key=lambda (k_, v_): k_)[:-1]:
            for n, i in enumerate(sorted(v)):
                for j in range(self.k):
                    n_child = sorted(self.d_nodes[k + 1])[j + (n * self.k)]
                    l_edges += [(self.network_nodes[i], self.network_nodes[n_child])]

        return l_edges

    def build_deterministic_io(self):
        ax_inputs = np.zeros((self.d, 1))
        ax_inputs[0, 0] = 1

        ax_outputs = None
        for k, v in sorted(self.d_nodes.items(), key=lambda (k_, v_): k_):
            ax = np.zeros((1, self.get_size(self.d, self.k)))
            ax[0, v] = 1
            if ax_outputs is not None:
                ax_outputs = np.vstack((ax_outputs, ax))
            else:
                ax_outputs = ax.copy()

        return ax_inputs, ax_outputs

    def build_random_io(self):

        # Generate random length
        n = np.random.randint(self.delay + 1, high=self.delay + 10)

        # Generate inpute and output arrays
        ax_inputs = np.zeros((n, 1))
        ax_outputs = np.zeros((n, self.get_size(self.d, self.k)))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, 1)), np.zeros((1, self.get_size(self.d, self.k)))

    def layout(self, ax_graph):

        ni, n, no = len(self.input_nodes), len(self.network_nodes), len(self.output_nodes)
        pos, pos_ = {}, nx.spring_layout(nx.from_numpy_array(ax_graph))

        pos.update({'inputs': {'pos': {i: pos_[i] for i in range(ni)}, 'color': 'r'}})
        pos.update({'networks': {'pos': {ni + i: pos_[ni + i] for i in range(n)}, 'color': 'k'}})
        pos.update({'outputs': {'pos': {ni + n + i: pos_[ni + n + i] for i in range(no)}, 'color': 'b'}})

        return pos


class AndPattern(TestPattern):
    """
    The primary test purpose of this pattern is to test for level convergence of network nodes. After a significant
    number of iteration, all edges should remain and only the level self.n (or higher) of network nodes should be activable
    """

    def __init__(self, n):

        self.n = n

        # Build list of nodess
        input_nodes = ['input_{}'.format(i) for i in range(n)]
        network_nodes = ['network_0']
        output_nodes = ['output_0']

        TestPattern.__init__(self, 'and', input_nodes, network_nodes, output_nodes, 2)

    def build_graph_pattern_final(self):

        # Build input edges
        l_edges = [(self.input_nodes[i], self.network_nodes[0]) for i in range(self.n)]

        # Build output edges
        l_edges += zip(self.network_nodes, self.output_nodes)

        return l_edges

    def build_graph_pattern_init(self):
        return self.build_graph_pattern_final()

    def build_deterministic_io(self):
        ax_inputs = np.ones((1, self.n))
        ax_outputs = np.ones((1, 1))

        return ax_inputs, ax_outputs

    def build_random_io(self):

        # Generate random 1 in inputs
        ax_one = np.random.choice(np.arange(self.n), size=np.random.randint(1, self.n), replace=False)

        ax_inputs = np.zeros((1, self.n))
        ax_inputs[0, ax_one] = 1

        # Generate outputs
        ax_outputs = np.zeros((1, 1))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.n)), np.zeros((1, 1))

    def layout(self):
        pos = dict()

        pos.update({'inputs': {'pos': {i: (0, i) for i in range(self.n)}, 'color': 'r'}})
        pos.update({'networks': {'pos': {self.n: (1, self.n / 2)}, 'color': 'k'}})
        pos.update({'outputs': {'pos': {self.n + 1: (2, self.n / 2)}, 'color': 'b'}})

        return pos


class XorPattern(TestPattern):
    """
    The primary test purpose of this pattern is to test for the removal of wrong edges from input to network nodes.
    After a significant number of iteration, self.na first input nodes should be linked to network node 0 and the
    self.nb following input nodes should be linked to network node 1, no other edges from inouts to network nodes should
    persist.
    """
    def __init__(self, na, nb):

        self.na, self.nb = na, nb

        # Build list of nodess
        input_nodes = ['input_{}'.format(i) for i in range(na + nb)]
        network_nodes = ['network_0', 'network_1']
        output_nodes = ['output_0', 'output_1']

        TestPattern.__init__(self, 'xor', input_nodes, network_nodes, output_nodes, 2)

    def build_graph_pattern_final(self):

        # Build input edges
        l_edges = [(self.input_nodes[i], self.network_nodes[0]) for i in range(self.na)]
        l_edges += [(self.input_nodes[i], self.network_nodes[1]) for i in range(self.na, self.na + self.nb)]

        # Build output edges
        l_edges += zip(self.network_nodes, self.output_nodes)

        return l_edges

    def build_graph_pattern_init(self):

        # Build input edges
        l_edges = [(self.input_nodes[i], self.network_nodes[0]) for i in range(self.na + self.nb)]
        l_edges += [(self.input_nodes[i], self.network_nodes[1]) for i in range(self.na + self.nb)]

        # Build output edges
        l_edges += zip(self.network_nodes, self.output_nodes)

        return l_edges

    def build_deterministic_io(self):
        ax_inputs_left = np.ones((1, self.na + self.nb))
        ax_inputs_left[0, self.na:] = 0

        ax_inputs_right = np.ones((1, self.na + self.nb))
        ax_inputs_right[0, :self.na] = 0

        ax_inputs = np.vstack((ax_inputs_left, ax_inputs_right))
        ax_outputs = np.array([[1, 0], [0, 1]])

        return ax_inputs, ax_outputs

    def build_random_io(self):

        # Generate random 1 in inputs
        ax_onea = np.random.choice(np.arange(self.na), size=np.random.randint(1, self.na), replace=False)
        ax_oneb = np.random.choice(np.arange(self.nb), size=np.random.randint(1, self.nb), replace=False)

        ax_inputs = np.zeros((1, self.na + self.nb))
        ax_inputs[0, np.hstack((ax_onea, self.na + ax_oneb))] = 1

        # Generate outputs
        ax_outputs = np.zeros((1, 2))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.na + self.nb)), np.zeros((1, 2))

    def layout(self):
        pos = dict()
        n = self.na + self.nb
        pos.update({'inputs': {'pos': {i: (0, i) for i in range(n)}, 'color': 'r'}})
        pos.update({'networks': {'pos': {n: (1, (n / 2) - (n / 4)), n + 1: (1, (n / 2) + (n / 4))}, 'color': 'k'}})
        pos.update({'outputs': {'pos': {n + 2: (2, (n / 2) - (n / 4)), n + 3: (2, (n / 2) + (n / 4))}, 'color': 'b'}})

        return pos


class RandomPattern(TestPattern):
    """
        random generation with the correcct convergence sentence
    """
    def __init__(self, ni, nd, no, delay, p, seed=666):

        self.ni, self.nd, self.no, self.delay, self.p, self.seed = ni, nd, no, delay, p, seed

        # Build list of nodess
        input_nodes = ['input_{}'.format(i) for i in range(self.ni)]
        network_nodes = ['network_{}'.format(i) for i in range(self.nd)]
        output_nodes = ['output_{}'.format(i) for i in range(self.no)]

        TestPattern.__init__(self, 'random', input_nodes, network_nodes, output_nodes, self.delay)

    def build_graph_pattern_final(self):
        raise NotImplementedError

    def build_graph_pattern_init(self, decrease='linear'):
        np.random.seed(self.seed)

        l_selected = [c for c in self.network_nodes if np.random.choice([0, 1], p=[1 - self.p, self.p]) == 1]
        l_candidates = [n for n in self.network_nodes if n not in l_selected]
        l_edges = [(np.random.choice(self.input_nodes), c) for c in l_selected]

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
            l_candidates = [n for n in self.network_nodes if n not in l_selected]

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
        ni, n, no = len(self.input_nodes), len(self.network_nodes), len(self.output_nodes)
        pos, pos_ = {}, nx.spring_layout(nx.from_numpy_array(ax_graph))

        pos.update({'inputs': {'pos': {i: pos_[i] for i in range(self.ni)}, 'color': 'r'}})
        pos.update({'networks': {'pos': {self.ni + i: pos_[self.ni + i] for i in range(self.nd)}, 'color': 'k'}})
        pos.update({'outputs': {'pos': {self.ni + self.nd + i: pos_[self.ni + self.nd + i] for i in range(self.no)},
                                'color': 'b'}})

        return pos



