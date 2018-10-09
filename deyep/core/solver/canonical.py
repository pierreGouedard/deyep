# Global import
from scipy.sparse import csc_matrix, vstack
import numpy as np

# Local import
from deyep.core.tools.equations.canonical import c_fnp, c_fcp, c_bop, c_bnp, c_bdu, c_bou, c_biu, c_buffer
from deyep.core.tools.equations.comon import bnt, bit, bcv, bcu
from deyep.core.solver.comon import DeepNetSolver
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.deep_network import DeepNetwork
driver = NumpyDriver()


class CanonicalDeepNetSolver(DeepNetSolver):

    def __init__(self, deep_network, delay, imputer, p0=1, verbose=1):
        DeepNetSolver.__init__(self, deep_network, delay, imputer, p0, 'canonical', verbose=verbose)

    def fit_epoch(self, n):

        i = 0
        while i < n:
            if self.t % 2 == 0:
                # Get new input and transmit forward
                self.sax_si = self.generate_input_signals(self.imputer.stream_next_forward(),
                                                          self.deep_network.input_nodes, self.dtype)
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
                self.sax_si = self.generate_input_signals(self.imputer.stream_next_forward(), self.deep_network.input_nodes,
                                                     self.dtype)
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

    def transform_input(self, imputer, n=int(1e6)):

        # reset signal just in case
        self.reset_forward_signals()

        ax_so, i = None, 0
        while n > i:

            # Get input signal
            sax_si = imputer.stream_next_forward()

            if sax_si is None:
                break

            # Transform and transmit forward
            self.sax_si = self.generate_input_signals(sax_si, self.deep_network.input_nodes, self.dtype)
            self.forward_transmiting()

            if ax_so is None:
                ax_so = self.sax_so.toarray().sum(axis=0) > 0
            else:
                ax_so = np.vstack((ax_so, self.sax_so.toarray().sum(axis=0) > 0))

            i += 1

        return ax_so

    def transform_array(self, ax_input, max_iter=10, return_activation=False):

        ax_input = np.vstack((ax_input, np.zeros([max_iter, ax_input.shape[-1]])))
        ax_activation = np.array([False] * len(self.deep_network.network_nodes), dtype=bool)

        # reset signal just in case
        self.reset_forward_signals()

        ax_so = None
        for ax_input_ in ax_input:

            # Get input signal
            sax_si = csc_matrix(ax_input_)

            # Transform and transmit forward
            self.sax_si = self.generate_input_signals(sax_si, self.deep_network.input_nodes, self.dtype)
            self.forward_transmiting()

            ax_activation_ = np.array(self.sax_sn.sum(axis=0) > 0)[0]
            if not ax_activation_.any():
                break

            ax_activation |= ax_activation_

            if ax_so is None:
                ax_so = self.sax_so.toarray().sum(axis=0) > 0
            else:
                ax_so = np.vstack((ax_so, self.sax_so.toarray().sum(axis=0) > 0))

        self.reset_forward_signals()

        if return_activation:
            return ax_activation, ax_so

        return ax_so

    def reverse_array(self, ax_output, max_iter=10, return_activation=False):
        ax_output = np.vstack((ax_output, np.zeros([max_iter, ax_output.shape[-1]])))
        ax_activation = np.array([False] * len(self.deep_network.network_nodes), dtype=bool)

        # reset signal just in case
        self.reset_backward_signals()

        ax_sib = None
        for ax_output_ in ax_output:

            # Init input signal
            sax_sob = csc_matrix(ax_output_[np.newaxis, :].repeat(self.deep_network.n_freq, axis=0))

            self.backward_transmit_simple(sax_sob)

            ax_activation_ = np.array(self.sax_snb.sum(axis=0) > 0)[0]

            if not ax_activation_.any():
                break

            ax_activation |= ax_activation_

            if ax_sib is None:
                ax_sib = self.sax_sib.toarray().sum(axis=0) > 0
            else:
                ax_sib = np.vstack((ax_sib, self.sax_sib.toarray().sum(axis=0) > 0))

        self.reset_backward_signals()

        if return_activation:
            return ax_activation, ax_sib

        return ax_sib

    def clean_network_nodes(self, max_iter=100, remove_non_firing_nodes=True):

        # Fetch inactive nodes: forward pass
        ax_input = np.ones(len(self.deep_network.input_nodes))
        ax_active_i, _ = self.transform_array(ax_input, max_iter=max_iter, return_activation=True)

        # Fetch inactive nodes: backward pass
        ax_output = np.ones(len(self.deep_network.output_nodes))
        ax_active_o, _ = self.reverse_array(ax_output, max_iter=max_iter, return_activation=True)

        # Get active nodes
        ax_active_nodes = ax_active_i & ax_active_o

        # Remove non firing nodes if necessary
        if remove_non_firing_nodes:
            ax_firing = self.deep_network.graph['N_f'] > 0
            ax_active_nodes &= ax_firing

        # Get cleaned network
        self.deep_network = DeepNetwork.reduce_network(self.deep_network, ax_active_nodes, self.basis)

        # Reset entirely the solver
        self.reset_solver(reset_imputer=True, reset_inputs=True, reset_time=True)

    def forward_processing(self, t, update_candidate=True):
        # transform signals
        self.sax_sn, self.ax_sa = c_fnp(self.sax_sn, self.deep_network.network_nodes, self.deep_network.tau, t)

        # Update candidate
        if update_candidate:
            self.deep_network.graph['N_f'] += self.ax_sa
            self.sax_C = c_fcp(self.ax_sa, self.deep_network.Cm, self.sax_C)

    def backward_transmiting(self, only_buffer=False):

        if not only_buffer:
            # Cache result of backward transmit
            sax_snb_ = bnt(self.deep_network.D, self.deep_network.O, self.sax_snb, self.sax_sob, self.sax_sab)
            sax_sib_ = bit(self.deep_network.I, self.sax_snb)

            # Update deep network structure
            ax_count = c_bdu(self.sax_snb, self.deep_network, penalty=self.p)
            ax_count += c_bou(self.sax_sob, self.sax_sab, self.deep_network, penalty=self.p)
            c_biu(self.sax_snb, self.deep_network, penalty=self.p)
            bcu(self.sax_Cb, self.deep_network, w0=self.deep_network.w0)

            self.deep_network.graph['N_r'] += ax_count * (ax_count > 0)
            self.deep_network.graph['N_p'] += - ax_count * (ax_count < 0)

            # Save result of backward transmit
            self.sax_snb = sax_snb_
            self.sax_sib = sax_sib_

        # Buffer forward signals
        self.sax_Cb, self.sax_sab, self.sax_sob = c_buffer(self.sax_C, self.ax_sa, len(self.deep_network.output_nodes),
                                                           self.sax_so)

    def backward_processing(self, sax_got, t):

        # Validate candidate
        self.sax_Cb = bcv(sax_got, self.sax_sob, self.sax_Cb)

        # Generate feedback
        self.sax_sob = c_bop(self.sax_sob, sax_got)

        # Backward network process
        self.sax_snb = c_bnp(self.deep_network.network_nodes, self.sax_snb, t, self.key_inputs)
