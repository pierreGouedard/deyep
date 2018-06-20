

# Global imports
import numpy as np

# local import


class TestPattern(object):
    delay = 2

    def __init__(self, name, input_nodes, network_nodes, output_nodes):

        self.name = name
        self.input_nodes = input_nodes
        self.network_nodes = network_nodes
        self.output_nodes = output_nodes

    def build_graph_pattern(self):
        raise NotImplementedError

    def build_deterministic_io(self):
        raise NotImplementedError

    def build_random_io(self):
        raise NotImplementedError

    def generate_io_sequence(self, length):
        raise NotImplementedError


class TreePattern(TestPattern):

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

        TestPattern.__init__(self, 'tree', input_nodes, network_nodes, output_nodes)

    @staticmethod
    def get_size(d, k):
        return sum([pow(k, i + 1) for i in range(d)])

    def build_graph_pattern(self):

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

    def build_deterministic_io(self):
        ax_inputs = np.zeros((self.d + self.delay, 1))
        ax_inputs[0, 0] = 1

        ax_outputs = np.zeros((self.delay, self.get_size(self.d, self.k)))
        for k, v in sorted(self.d_nodes.items(), key=lambda (k_, v_): k_):
            ax = np.zeros((1, self.get_size(self.d, self.k)))
            ax[0, v] = 1

            ax_outputs = np.vstack((ax_outputs, ax))

        return ax_inputs, ax_outputs

    def build_random_io(self):

        # Generate random length
        n = np.random.randint(self.delay + 1, high=self.delay + 10)

        # Generate inpute and output arrays
        ax_inputs = np.zeros((n, 1))
        ax_outputs = np.zeros((n, self.get_size(self.d, self.k)))

        return ax_inputs, ax_outputs

    def generate_io_sequence(self, length):
        ax_isequence = np.zeros((1, 1))
        ax_osequence = np.zeros((1, self.get_size(self.d, self.k)))

        for _ in range(length):

            # Add deterministic pattern
            ax_inputs, ax_outputs = self.build_deterministic_io()
            ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

            # Add random pattern
            ax_inputs, ax_outputs = self.build_random_io()
            ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

        return ax_isequence, ax_osequence


class AndPattern(TestPattern):

    def __init__(self, n):

        self.n = n

        # Build list of nodess
        input_nodes = ['input_{}'.format(i) for i in range(n)]
        network_nodes = ['network_0']
        output_nodes = ['output_0']

        TestPattern.__init__(self, 'and', input_nodes, network_nodes, output_nodes)

    def build_graph_pattern(self):

        # Build input edges
        l_edges = [(self.input_nodes[i], self.network_nodes[0]) for i in range(self.n)]

        # Build output edges
        l_edges += [zip(self.network_nodes, self.output_nodes)]

        return l_edges

    def build_deterministic_io(self):
        ax_inputs = np.ones((1, self.n))
        ax_inputs = np.vstack((ax_inputs, np.zeros((self.delay, self.n))))

        ax_outputs = np.zeros((self.delay, 1))
        ax_outputs = np.vstack((ax_outputs, np.ones((1, 1))))

        return ax_inputs, ax_outputs

    def build_random_io(self):

        # Generate random 1 in inputs
        ax_one = np.random.choice(np.arange(self.n), size=np.random.randint(1, self.n - 1), replace=False)

        ax_inputs = np.zeros((1, self.n))
        ax_inputs[0, ax_one] = 1

        # Generate outputs
        ax_outputs = np.zeros((1, 1))

        return ax_inputs, ax_outputs

    def generate_io_sequence(self, length):
        ax_isequence = np.zeros((1, self.n))
        ax_osequence = np.zeros((1, 1))

        for _ in range(length):

            # Add deterministic pattern
            ax_inputs, ax_outputs = self.build_deterministic_io()
            ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

            # Add random pattern
            ax_inputs, ax_outputs = self.build_random_io()
            ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

        return ax_isequence, ax_osequence


class XorPattern(TestPattern):

    def __init__(self, na, nb):

        self.na, self.nb = na, nb

        # Build list of nodess
        input_nodes = ['input_{}'.format(i) for i in range(na + nb)]
        network_nodes = ['network_0', 'network_1']
        output_nodes = ['output_0', 'output_1']

        TestPattern.__init__(self, 'xor', input_nodes, network_nodes, output_nodes)

    def build_graph_pattern(self):

        # Build input edges
        l_edges = [(self.input_nodes[i], self.network_nodes[0]) for i in range(self.na + self.nb)]
        l_edges += [(self.input_nodes[i], self.network_nodes[1]) for i in range(self.na + self.nb)]

        # Build output edges
        l_edges += [zip(self.network_nodes, self.output_nodes)]

        return l_edges

    def build_deterministic_io(self):
        ax_inputs_left = np.ones((1, self.na + self.nb))
        ax_inputs_left[0, self.na:] = 0

        ax_inputs_right = np.ones((1, self.na + self.nb))
        ax_inputs_right[0, :self.na] = 0

        ax_inputs = np.vstack((ax_inputs_left, ax_inputs_right, np.zeros((self.delay, self.na + self.nb))))
        ax_outputs = np.vstack((np.zeros((self.delay, 2)), np.array([[1, 0], [0, 1]])))

        return ax_inputs, ax_outputs

    def build_random_io(self):

        # Generate random 1 in inputs
        ax_onea = np.random.choice(np.arange(self.na), size=np.random.randint(1, self.na - 1), replace=False)
        ax_oneb = np.random.choice(np.arange(self.nb), size=np.random.randint(1, self.nb - 1), replace=False)

        ax_inputs = np.zeros((1, self.na + self.nb))
        ax_inputs[0, np.hstack((ax_onea, ax_oneb))] = 1

        # Generate outputs
        ax_outputs = np.zeros((1, 1))

        return ax_inputs, ax_outputs

    def generate_io_sequence(self, length):
        ax_isequence = np.zeros((1, self.na + self.nb))
        ax_osequence = np.zeros((1, 1))

        for _ in range(length):

            # Add deterministic pattern
            ax_inputs, ax_outputs = self.build_deterministic_io()
            ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

            # Add random pattern
            ax_inputs, ax_outputs = self.build_random_io()
            ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

        return ax_isequence, ax_osequence