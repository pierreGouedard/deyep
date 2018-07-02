# Global import
from scipy.sparse import csc_matrix
import numpy as np


class DeepNetSolver(object):
    def __init__(self, deep_network, delay, imputer):

        self.deep_network = deep_network
        self.imputer = imputer
        self.delay = delay
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, self.C = init_core_forward_signal(deep_network)
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab, self.Cb = init_core_forward_signal(deep_network)

        self.t = 0

    def run_epoch(self, n):
        i = 0
        while i <= n:
            if self.t % 2 == 0:
                self.sax_si = self.imputer.get_next_input()
                self.forward_transmiting()

                if self.t / 2 >= self.delay:
                    sax_got = self.imputer.get_next_output()
                    self.backward_processing(sax_got)
                i += 1

            else:
                self.forward_processing()
                self.backward_transmiting()

            self.t += 1

    def forward_transmiting(self):
        # Transmit fresh input signal and residual network signal
        raise NotImplementedError

    def forward_processing(self):
        # transform signals

        # Update candidate

        raise NotImplementedError

    def backward_transmiting(self):

        # Buffer forward signals

        # Transmit backward

        # update structure
        raise NotImplementedError

    def backward_processing(self, sax_got):
        # Get feedback

        # Validate candidate

        # Update network

        raise NotImplementedError


def init_core_forward_signal(dn):

    N = dn.n_freq
    sax_si = csc_matrix((N, len(dn.input_nodes)))
    sax_sn = csc_matrix((N, len(dn.network_nodes)))
    sax_so = csc_matrix((N, len(dn.output_nodes)))
    ax_sa = np.array([False] * len(dn.network_nodes))
    sax_C = csc_matrix((len(dn.network_nodes), len(dn.network_nodes)))

    return sax_si, sax_sn, sax_so, ax_sa, sax_C

def init_core_backward_signal(dn):
    N = dn.n_freq

    sax_sib = csc_matrix((N, len(dn.input_nodes)))
    sax_snb = csc_matrix((N, len(dn.network_nodes)))
    sax_sob = csc_matrix((N, len(dn.output_nodes)))
    sax_sab = csc_matrix(np.array([[False] * len(dn.network_nodes)]).repeat(len(dn.output_nodes), axis=0))
    sax_Cb = csc_matrix((len(dn.network_nodes), len(dn.network_nodes)))

    return sax_sib, sax_snb, sax_sob, sax_sab, sax_Cb



