# Global import
from scipy.sparse import csc_matrix, lil_matrix, vstack, hstack
from deyep.core.firing_graph.network_dry import DeepNetworkDry
import numpy as np

# Local import


class DeepNetRunner(object):
    def __init__(self, deep_network, delay, imputer, verbose=0):
        # Core params
        self.delay = delay
        self.verbose = verbose

        # Data structure
        self.deep_network = DeepNetworkDry.from_deep_network(deep_network)
        self.imputer = imputer

    def reset_runner(self):
        self.imputer.stream_features(is_cyclic=False)
        return self

    def transform_array(self, l_nodes=None):

        sax_out, sax_sn, stop, i, d_ = None, csc_matrix((1, self.deep_network.D.shape[0]), dtype=int), False, 0, 0
        while not stop:
            # Get input signal
            sax_si = self.imputer.stream_next_forward()

            if sax_si is None:
                d_ += 1
                sax_si = csc_matrix((1, self.deep_network.I.shape[0]), dtype=int)
                if d_ >= self.delay:
                    stop = True
                    continue

            # transmit forward
            sax_sn_, sax_so = DeepNetRunner.forward_transmit(
                self.deep_network.D, self.deep_network.I, self.deep_network.O, sax_si, sax_sn
            )

            # Save outputs
            if sax_out is None:
                sax_out = sax_so > 0
            else:
                sax_out = vstack([sax_out, sax_so > 0], format='csc')

            # Compute activations
            sax_sn = DeepNetRunner.forward_activation(sax_sn_, self.deep_network.network_nodes)

            i += 1

        if l_nodes is not None:
            return sax_out[self.delay - 1:, :][:, l_nodes], {k: i for i, k in enumerate(l_nodes)}

        return sax_out[self.delay - 1:, :]

    def track_activity(self, delay, l_nodes=None):

        sax_activation, sax_sn, stop, d_ = None, csc_matrix((1, self.deep_network.D.shape[0]), dtype=int), False, 0

        while not stop:
            # Get input signal
            sax_si = self.imputer.stream_next_forward()

            if sax_si is None:
                d_ += 1
                sax_si = csc_matrix((1, self.deep_network.I.shape[0]), dtype=int)
                if d_ >= delay:
                    stop = True
                    continue

            # transmit forward
            sax_sn_, _ = DeepNetRunner.forward_transmit(
                self.deep_network.D, self.deep_network.I, self.deep_network.O, sax_si, sax_sn
            )

            # Compute activations
            sax_sn = DeepNetRunner.forward_activation(sax_sn_, self.deep_network.network_nodes, get_level=True)

            # Save activation
            if sax_activation is None:
                sax_activation = sax_sn
            else:
                sax_activation = vstack([sax_activation, sax_sn], format='csc')

            sax_sn = (sax_sn > 0).astype(int)

        if l_nodes is not None:
            return sax_activation[delay - 1:, :][:, l_nodes], {k: i for i, k in enumerate(l_nodes)}

        return sax_activation[delay - 1:, :]

    @staticmethod
    def forward_transmit(sax_D, sax_I, sax_O, sax_si, sax_sn):
        sax_sn_ = sax_sn.dot(sax_D) + sax_si.dot(sax_I)
        sax_so = sax_sn.dot(sax_O)

        return sax_sn_, sax_so

    @staticmethod
    def forward_activation(sax_sn, l_nodes, get_level=False):
        sax_sn_ = lil_matrix(sax_sn.shape, dtype=int)

        # activate node signal iff level is allowed
        for i in np.unique(sax_sn.nonzero()[1]):
            if l_nodes[i].d_levels[sax_sn[0, i]]:
                if get_level:
                    sax_sn_[0, i] = sax_sn[0, i]
                else:
                    sax_sn_[0, i] = 1

        return sax_sn_.tocsc()

    def reverse_array(self, ax_output, max_iter=10):
        raise NotImplementedError

    def backward_transmit(self):
        raise NotImplementedError

