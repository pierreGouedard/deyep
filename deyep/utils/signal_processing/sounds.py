# global import
from scipy import signal
import numpy as np

# Local import


def butter_lowpass(cutoff, fs, order):
    """
    Compute the coefficients of butter low pass polynome
    :param cutoff: float cut-off frequency in Hertz
    :param fs: float sampling frequecy in Herts
    :param order: int order of the butter polynome
    :return: oefficient of the butter polynome
    """

    # Get nyquist frequency and cut-off in rad/s
    normal_cutoff = float(cutoff * 2) / fs

    # Get coeff of polynome
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    return b, a


def butter_lowpass_filter(s, cutOff, fs, order=20):
    """

    :param data:
    :param cutOff:
    :param fs:
    :param order:
    :return:
    """
    # Compute the coefficient
    b, a = butter_lowpass(cutOff, fs, order=order)

    # low pass input signal
    s_ = signal.lfilter(b, a, s)

    return s_


def compute_stft_decomposition(s, n, fs, p, fm, no, ns):
    """

    :param s:
    :param n:
    :param fs:
    :param p:
    :param fm:
    :param no:
    :param ns:
    :return:
    """

    # Divide the signal into pieces of n secondes, with 0.5 overlap
    if len(s) / (fs * n) > 0:
        r = int(np.ceil(float(len(s)) / (fs * n)))
        d_sig = {'part_{}'.format(i): dict(zip(['signal', 'mask'], get_part(i, s, n, fs, p))) for i in range(r)}
    else:
        d_sig = {'part_0': s, 'mask': [True]*len(s)}

    d_stft = dict()
    for name, d_part in d_sig.items():
        import IPython
        IPython.embed()

        # Compute stft
        f, t, Zxx = signal.stft(d_part['signal'], fs, noverlap=no, nperseg=ns, boundary=None)

        # Build time mask from signal mask
        time_mask, time_offset = transform_mask(d_part['mask'], int(name.split('_')[-1]), n, t, ns, fs)

        # Store result in dictonnary
        d_stft.update({name: {'frequencies': f[f < fm], 'time': t[time_mask], 'offset': time_offset,
                              'imaginary': np.imag(Zxx[f < fm][:, time_mask]),
                              'real': np.real(Zxx[f < fm][:, time_mask])}})

    return d_stft


def transform_mask(mask, i, n, t, ns, fs):

    new_mask = []
    for x in t:
        new_mask += [all(mask[int((x * fs) - (ns / 2)): int((x * fs) + (ns / 2))])]

    return np.array(new_mask), i * n


def optimize_segmentation(ns, n, p, fs):
    """
    make sure that the

    :param ns:
    :param n:
    :param p:
    :param fs:
    :return:
    """

    # Test if ns is a comon divisor of n * p * fs and n * fs
    r = int(n * p * fs / ns)

    if r == float(n * p * fs) / ns:
        return ns

    else:
        while float(n * p * fs) / r != int(n * p * fs) / r:
            r += 1

        ns_ = int(n * p * fs) / r

        return ns_


def inverse_stft_decomposition(d_stft, fs, no, ns):
    """

    :param Zxx:
    :param fs:
    :param no:
    :param ns:
    :return:
    """
    import IPython
    IPython.embed()
    # Recover the signal from a
    l_stft = sorted(d_stft.items(), key=lambda x: int(x[0].split('_')[-1]))
    Z_xx = np.hstack((x['real'] + x['imaginary'] * np.complex(0, 1) for _, x in l_stft))
    # TODO:
    # First we need to padd along the freq axis so that Z_xx.shape[-2] - 1 = nperseg / 2 (which was the initial length of Z_xx before mask
    # May be we should add a field in dictonarry with : frequency to padd !!!!!!!!
    _, s_rec = signal.istft(Z_xx, fs=fs, nperseg=ns, noverlap=no, boundary=None)

    return s_rec


def get_part(i, s, n, fs, p):
    """

    :param i: int iteration
    :param s: array_like signal that is being decomposed
    :param n: int duration (in seconds) of part in the signal
    :param fs: float sampling frequency
    :param p: float overlapping proportions
    :return: ith part of signal s

    """
    assert p < 1.0, 'the overlap parameter p should be loawer than 1.0, instead {}'.format(p)

    s_ = s[max(int((i * n * fs) - (n * p * fs)), 0): min(int(((i + 1) * n * fs) + (n * p * fs)), len(s))]

    if i == 0:
        mask = np.array([True] * int(n * fs) + [False] * int(n * p * fs))

    elif (i + 1) * n * fs >= len(s):
        mask = np.array([False] * int(n * p * fs) + [True] * len(s_[int(n * p * fs):]))

    elif ((i + 1) * n * fs) + (n * p * fs) >= len(s):
        mask = np.array([False] * int(n * p * fs) + [True] * int(n * fs) + [False] * len(s_[int(n * (p + 1) * fs):]))

    else:
        mask = np.array([False] * int(n * p * fs) + [True] * int(n * fs) + [False] * int(n * p * fs))

    return [s_, mask]


