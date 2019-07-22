# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.utils.driver.nmp import NumpyDriver


class TestSignal(object):
    driver = NumpyDriver()

    def __init__(self, name):
        self.name = name
        self.imputer = None

    @property
    def target_norm(self):
        raise NotImplementedError

    def phi(self, nu):
        raise NotImplementedError

    def nu(self, i):
        raise NotImplementedError

    def rho(self, i):
        raise NotImplementedError

    def T(self, t, i):
        raise NotImplementedError

    def N(self, t, i):
        raise NotImplementedError

    def get_tuple(self, i, t):
        raise NotImplementedError

    def generate_io_sequence(self, n):
        raise NotImplementedError

    def stream_io_sequence(self, n):
        raise NotImplementedError

    def create_imputer(self, sax_in, sax_out, return_dirs=False):

        dirname = 'tmp_{}_'.format(self.name)
        tmpdiri = self.driver.TempDir(dirname, suffix='in', create=True)
        tmpdiro = self.driver.TempDir(dirname, suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        self.driver.write_file(sax_in, self.driver.join(tmpdiri.path, 'forward.npz'), is_sparse=True)
        self.driver.write_file(sax_out, self.driver.join(tmpdiri.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleArrayImputer('test', tmpdiri.path, tmpdiro.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        if return_dirs:
            return imputer, tmpdiri, tmpdiro

        tmpdiri.remove(), tmpdiro.remove()

        return imputer


class SignalPlusNoise(TestSignal):

    def __init__(self, n_sim, n_bits, p_target, n_targets, p_noise):

        # Size of the simulation
        self.n_sim, self.n_bits, self.n_targets = n_sim, n_bits,n_targets

        # Base param of simulation
        self.p_target, self.p_noise = p_target, p_noise

        self.target_bits = [np.random.choice(range(self.n_bits), n_targets, replace=False) for _ in range(self.n_sim)]

        TestSignal.__init__(self, 'SignalPlusNoise')

    @property
    def target_norm(self):
        return self.p_target + ((1 - self.p_target) * pow(self.p_noise, self.n_targets))

    def phi(self, nu):
        return (self.p_target / (self.p_target + nu))

    def nu(self, i):
        return (1 - self.p_target) * (1 + self.p_noise) * pow(self.p_noise, i)

    def rho(self, i):
        return np.ceil(self.phi(self.nu(i)) / (1 - self.phi(self.nu(i))))

    def T(self, t, i):
        return t / (self.p_target + self.nu(i))

    def N(self, t, i):
        return - int(t * (self.phi(self.nu(i)) * (self.rho(i) + 1) - self.rho(i)))

    def get_tuple(self, i, t):
        return self.nu(i), self.rho(i), self.T(t, i), self.N(t, i)

    def generate_io_sequence(self, n):

        ax_inoise, ax_onoise = self.generate_noise_sequence(n)
        ax_itarget, ax_otarget = self.generate_target_sequence(n)

        ax_inputs = ax_inoise + ax_itarget > 0
        ax_outputs = ax_onoise + ax_otarget > 0

        return csc_matrix(ax_inputs, dtype=bool), csc_matrix(ax_outputs, dtype=bool)

    def stream_io_sequence(self, n, return_dirs=True):
        sax_in, sax_out = self.generate_io_sequence(n)

        if return_dirs:
            self.imputer, tmpdiri, tmpdiro = self.create_imputer(sax_in, sax_out, return_dirs=return_dirs)
            return self.imputer, tmpdiri, tmpdiro

        else:
            self.imputer = self.create_imputer(sax_in, sax_out)

        return self.imputer

    def generate_noise_sequence(self, n):
        # Init noisy sequence
        ax_outputs = np.zeros((n, self.n_sim))
        ax_inputs = np.random.binomial(1, self.p_noise, (n, self.n_bits * self.n_sim))

        # Activate whenever every target are active
        for i in range(n):
            # Build next output
            for k, l_indices in enumerate(self.target_bits):
                if ax_inputs[i, l_indices].all():
                    ax_outputs[i, k] = 1

        return ax_inputs, ax_outputs

    def generate_target_sequence(self, n):
        ax_inputs, ax_outputs = np.zeros((n, self.n_bits * self.n_sim)), np.zeros((n, self.n_sim))
        ax_activations = np.random.binomial(1, self.p_target, (n, self.n_sim))

        for i in range(n):
            for j in range(self.n_sim):
                if ax_activations[i, j] == 1:
                    ax_inputs[i, self.target_bits[j]], ax_outputs[i, j] = 1, 1

        return ax_inputs, ax_outputs