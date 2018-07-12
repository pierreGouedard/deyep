# Global import
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_spectogram(Zxx, f, t):
    """
    Interactive plot of spectogram
    :param Zxx: array-like result of the stft
    :param f: array-like frequency range
    :param t: array-like time range
    :return:
    """
    import IPython
    IPython.embed()

    # run code below
    Sxx = np.abs(Zxx)

    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_signal(s, t):
    import IPython
    IPython.embed()

    plt.plot(t, s)
    plt.ylabel('value')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_signals(l_s, t):
    import IPython
    IPython.embed()

    fig = plt.figure()
    for s in l_s:
        plt.plot(t, s)

    plt.ylabel('value')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_fft(N, xf, yf):
    """
    plot frequencies
    :param N: int len of signal / fft
    :param xf: array-like frequencies (label of x axis)
    :param yf: frequency power
    :return:
    """
    import IPython
    IPython.embed()

    plt.plot(xf, 1.0 / N * np.abs(yf))
    plt.grid()
    plt.show()


def plot_graph(ax_graph):
    """
    plot simple graph
    :param ax_graph: numpy.ndarray
    :return:
    """
    nx_graph = nx.from_numpy_array(ax_graph)
    import IPython
    IPython.embed()
    nx.draw(nx_graph)
    plt.show()


