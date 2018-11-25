# Global import
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from deyep.core.tools.equations.backward import bop, bnp, bnt, bit, bcv, buffer
from deyep.core.tools.equations.forward import fnp, fcp, fot, fnt
from deyep.core.tools.equations.structure import bdu, bou, biu, bcu


class DeepNetSolver(object):
    def __init__(self, deep_network, delay, imputer, p0, verbose=0):

        # Core params
        self.p = p0
        self.delay = delay
        self.verbose = verbose

        # Data structure
        self.deep_network = deep_network
        self.imputer = imputer
        self.key_inputs = set(['N={},k={}'.format(n.basis.N, n.basis.key) for n in self.deep_network.input_nodes])

        # Init signals
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, self.sax_C = \
            DeepNetSolver.init_core_forward_signal(deep_network)
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab, self.sax_Cb = \
            DeepNetSolver.init_core_backward_signal(deep_network)

        self.t = 0
        self.t_fp = 0
        self.transformed = False

    def reset_solver(self, reset_imputer=False, reset_time=False, reset_inputs=False):
        if reset_imputer:
            self.imputer.stream_features()

        if reset_time:
            self.t, self.t_fp = 0, 0

        if reset_inputs:
            self.key_inputs = set(['N={},k={}'.format(n.basis.N, n.basis.key) for n in self.deep_network.input_nodes])

        # reset signals
        self.reset_forward_signals()
        self.reset_backward_signals()

    def reset_forward_signals(self):
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, self.sax_C = \
            DeepNetSolver.init_core_forward_signal(self.deep_network)

    def reset_backward_signals(self):
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab, self.sax_Cb = \
            DeepNetSolver.init_core_backward_signal(self.deep_network)

    def fit_epoch(self, n):

        i = 0
        while i < n:
            if self.t % 2 == 0:
                # Get new input and transmit forward
                self.sax_si = self.generate_input_signals(self.imputer.stream_next_forward(),
                                                          self.deep_network.input_nodes)
                self.forward_transmiting()

                # Run backward processing if delay is reached
                if self.t / 2 >= self.delay:
                    sax_got = self.imputer.stream_next_backward()
                    self.backward_processing(sax_got, self.t_fp - 1)

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    self.backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)

                # Run forward processing
                self.forward_processing(self.t_fp)
                self.t_fp += 1
                i += 1

            self.t += 1

            if self.verbose == 1 and self.t % 100 == 0:
                print '[Info]: iteration {} completed'.format(self.t, n)

    def epoch_analysis(self, n):

        import time
        l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt, i = [], [], [], [], [], 0

        while i < n:
            t = time.time()
            if self.t % 2 == 0:
                # Get new input and transmit forward
                t0 = time.time()
                self.sax_si = self.generate_input_signals(self.imputer.stream_next_forward(),
                                                          self.deep_network.input_nodes,)
                self.forward_transmiting()
                l_t_forwardt += [time.time() - t0]

                # Run backward processing if delay is reached
                if self.t / 2 >= self.delay:
                    t0 = time.time()
                    sax_got = self.imputer.stream_next_backward()
                    self.backward_processing(sax_got, self.t_fp - 1)
                    l_t_backwardp += [time.time() - t0]

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    t0 = time.time()
                    self.backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)
                    l_t_backwardt += [time.time() - t0]

                # Run forward processing
                t0 = time.time()
                self.forward_processing(self.t_fp)
                l_t_forwardp += [time.time() - t0]
                self.t_fp += 1
                i += 1

            self.t += 1
            l_t_epoch += [time.time() - t]

        return l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt

    def forward_transmiting(self):
        # Output transmit
        self.sax_so = fot(self.deep_network.O, self.sax_sn)

        # network transmit
        self.sax_sn = fnt(self.deep_network.D, self.deep_network.I, self.sax_sn, self.sax_si)

    def forward_processing(self, t, update_candidate=True):
        # transform signals
        self.sax_sn, self.ax_sa = fnp(self.sax_sn, self.deep_network.network_nodes, self.deep_network.tau, t)

        # Update candidate
        if update_candidate:
            self.deep_network.graph['N_f'] += self.ax_sa
            self.sax_C = fcp(self.ax_sa, self.deep_network.Cm)

    def backward_transmiting(self, only_buffer=False):

        if not only_buffer:
            # Cache result of backward transmit
            sax_snb_ = bnt(self.deep_network.D, self.deep_network.O, self.sax_snb, self.sax_sob, self.sax_sab)
            sax_sib_ = bit(self.deep_network.I, self.sax_snb)

            # Update deep network structure
            ax_count = bdu(self.sax_snb, self.deep_network, penalty=self.p)
            ax_count += bou(self.sax_sob, self.sax_sab, self.deep_network, penalty=self.p)
            biu(self.sax_snb, self.deep_network, penalty=self.p)
            bcu(self.sax_Cb, self.deep_network, w0=self.deep_network.w0)

            self.deep_network.graph['N_r'] += ax_count * (ax_count > 0)
            self.deep_network.graph['N_p'] += - ax_count * (ax_count < 0)

            # Save result of backward transmit
            self.sax_snb = sax_snb_
            self.sax_sib = sax_sib_

        # Buffer forward signals
        self.sax_Cb, self.sax_sab, self.sax_sob = buffer(self.sax_C, self.ax_sa, len(self.deep_network.output_nodes),
                                                         self.sax_so)

    def backward_processing(self, sax_got, t):

        # Validate candidate
        self.sax_Cb = bcv(sax_got, self.sax_sob, self.sax_Cb)

        # Generate feedback
        self.sax_sob = bop(self.sax_sob, sax_got)

        # Backward network process
        self.sax_snb = bnp(self.deep_network.network_nodes, self.sax_snb, t, self.key_inputs)

    @staticmethod
    def generate_input_signals(sax_i, input_nodes):
        sax_si = csc_matrix((len(input_nodes), input_nodes[0].basis.N), dtype=int)

        for i,  n in enumerate(input_nodes):
            if sax_i[0, i] >= 1:
                sax_si[i, :] = n.basis.base

        return sax_si.transpose()

    @staticmethod
    def init_core_forward_signal(dn):

        N = dn.n_freq
        sax_si = csc_matrix((N, len(dn.input_nodes)), dtype=int)
        sax_sn = csc_matrix((N, len(dn.network_nodes)), dtype=int)
        sax_so = csc_matrix((N, len(dn.output_nodes)), dtype=int)
        ax_sa = np.array([False] * len(dn.network_nodes))
        sax_C = csc_matrix((len(dn.network_nodes), len(dn.output_nodes)))

        return sax_si, sax_sn, sax_so, ax_sa, sax_C

    @staticmethod
    def init_core_backward_signal(dn):
        N = dn.n_freq

        sax_sib = csc_matrix((N, len(dn.input_nodes)), dtype=int)
        sax_snb = csc_matrix((N, len(dn.network_nodes)), dtype=int)
        sax_sob = csc_matrix((N, len(dn.output_nodes)), dtype=int)
        sax_sab = csc_matrix(np.array([[False] * len(dn.network_nodes)]).repeat(len(dn.output_nodes), axis=0))
        sax_Cb = csc_matrix((len(dn.network_nodes), len(dn.output_nodes)))

        return sax_sib, sax_snb, sax_sob, sax_sab, sax_Cb
