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
    ).sample().build_graph_multiple_output()

    # Drain
    drn = drainer.FiringGraphDrainer(t, simu.rho(i), resolution, smp.firing_graph.copy(), imputer, verbose=1)

    # init tracking of simulation
    l_ps_bits, l_target_bits = list(smp.preselect_bits[0]), list(simu.target_bits[0])
    l_noisy_bits = list(set(l_ps_bits).difference(l_target_bits))
    ax_noisy_bits, ax_target_bits = np.zeros((len(l_noisy_bits), 1)), np.zeros((len(l_target_bits), 1))

    # Print some general information on simulation
    if verbose > 0:
        print('simulation parameter: i={}, t={}, N={}, mean_score={}'.format(
            i, t, simu.N(t, i), simu.mean_score_signal(t, i)
        ))

    stop, j, fg = False, 0, resolution
    while not stop:
        print('Iteration {}'.format((j + 1) * resolution))

        for _ in range(int(resolution / drn.bs)):
            fg = drn.drain().firing_graph

        ax_noisy_bits = np.hstack((ax_noisy_bits, fg.Iw.toarray()[l_noisy_bits, 0][:, np.newaxis]))
        ax_target_bits = np.hstack((ax_target_bits, fg.Iw.toarray()[l_target_bits, 0][:, np.newaxis]))

        if not fg.Im.toarray().any():
            stop = True
            continue

        batch_size = max(min(t - fg.backward_firing['i'].toarray().max(), resolution), 1)
        drn.bs = int(resolution / np.ceil(resolution / batch_size))
        print('batch_size: {}'.format(drn.bs))
        drn.reset_all()
        j += 1

    from deyep.utils import interactive_plots as ip

    import IPython
    IPython.embed()

    ip.multi_line_plot_colored(
        {'b': ax_target_bits[:, 1:], 'r': ax_noisy_bits[:, 1:],
         'k': np.ones((1, ax_target_bits.shape[1])) * simu.mean_score_signal(t, 0)},
        title=r'Score processes ($p_f=0.3$, $p_n=0.3$, $t=500$) ',
        ylab=r'Value of the score process',
        xlab=r"Number of iteration ($\times 10$)"
    )

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