# Global import
import numpy as np

# Local import
from deyep.utils import test_signal as ts
from deyep.core.solver import sampler, drainer


def run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, i, resolution=10, verbose=0):
    """

    :param t:
    :param n_bits:
    :param p_noise:
    :param p_target:
    :param n_targets:
    :param i:
    :return:
    """

    # Create simulation and get imputer
    simu = ts.SignalPlusNoise(1, n_bits, p_target, n_targets, p_noise)
    imputer, dirin, dirout = simu.stream_io_sequence(10000, return_dirs=True)

    if i > 0:
        target_bits = {0: np.random.choice(simu.target_bits[0], i, replace=False)}
    else:
        target_bits = None

    # Sample and drain
    smp = sampler.Sampler(
        [simu.n_sim * simu.n_bits, simu.n_sim], simu.N(t, i), imputer, selected_bits=target_bits
    ).sample().build_graph_multiple_output(t)

    # Drain
    drn = drainer.FiringGraphDrainer(
        simu.rho(i), smp.firing_graph.copy(), imputer, depth=2 if i == 0 else 3, verbose=1
    )

    # init tracking of simulation
    l_ps_bits, l_target_bits = list(smp.preselect_bits[0]), list(simu.target_bits[0])
    l_noisy_bits = list(set(l_ps_bits).difference(l_target_bits))
    ax_noisy_bits, ax_target_bits = np.zeros((len(l_noisy_bits), 1)), np.zeros((len(l_target_bits), 1))

    # Print some general information on simulation
    if verbose > 0:
        print 'simulation parameter: i={}, t={}, N={}, mean_score={}'.format(
            i, t, simu.N(t, i), simu.mean_score_signal(t, i)
        )
        sax_in, sax_out = imputer.features_forward.toarray()[:500], imputer.features_backward[:500]

    stop, j, fg = False, 0, None
    while not stop:
        print 'Iteration {}'.format((j + 1) * resolution)
        fg = drn.drain(resolution, early_stoping=True).firing_graph
        ax_noisy_bits = np.hstack((ax_noisy_bits, fg.Iw.toarray()[l_noisy_bits, 0][:, np.newaxis]))
        ax_target_bits = np.hstack((ax_target_bits, fg.Iw.toarray()[l_target_bits, 0][:, np.newaxis]))

        stop = not fg.mask_mat['I'].toarray().any()
        j += 1

    # Remove tmpdir
    dirin.remove(), dirout.remove()

    return ax_noisy_bits, ax_target_bits, simu


def run_sparse_simulation(t, n_bits, p_noise, p_target, n_targets, i, resolution=10):
    """

    :param t:
    :param n_bits:
    :param p_noise:
    :param p_target:
    :param n_targets:
    :param i:
    :return:
    """
    raise NotImplemented