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


def butter_lowpass_filter(s, cutOff, fs, order=50):
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
        # Compute stft
        f, t, Zxx = signal.stft(d_part['signal'], fs=fs, window='boxcar', noverlap=no, nperseg=ns, boundary=None)

        # Build time mask from signal mask
        time_mask, time_offset = transform_mask(d_part['mask'], int(name.split('_')[-1]), n, t, ns, fs)

        # Store result in dictonnary we only need
        d_stft.update({name: {'freq': f[f < fm], 'time': t[time_mask], 'offset': time_offset,
                              'im': np.imag(Zxx[f < fm][:, time_mask]), 're': np.real(Zxx[f < fm][:, time_mask]),
                              'window': Zxx[f >= fm][:, time_mask]}})

    return d_stft


def transform_mask(mask, i, n, t, ns, fs):

    new_mask = []
    for x in t:
        new_mask += [all(mask[int((x * fs) - (ns / 2)): int((x * fs) + (ns / 2))])]

    return np.array(new_mask), i * n


def optimize_segmentation(ns, n, fs):
    """
    make sure that the

    :param ns:
    :param n:
    :param p:
    :param fs:
    :return:
    """

    # Test if ns is a comon divisor of n * p * fs and n * fs
    r = int(n * fs / ns)

    if r == float(n * fs) / ns:
        return ns

    else:
        while float(n * fs) / r != int(n * fs) / r:
            r += 1

        ns_ = int(n * fs) / r

        return ns_


def inverse_stft_decomposition(d_stft, fs, no, ns, noise=0):
    """

    :param Zxx:
    :param fs:
    :param no:
    :param ns:
    :return:
    """


    # Recover the signal from a
    l_stft, Z_xx = sorted(d_stft.items(), key=lambda x: int(x[0].split('_')[-1])), None

    # re build the entire signal
    Z_xx = np.hstack((np.vstack((x['re'] + (x['im'] * np.complex(0, 1)), x['window'])) for _, x in l_stft))

    if noise is not None:
        Z_xx += noise*np.random.randn(Z_xx.shape[0], Z_xx.shape[1]) + \
                noise*(np.random.randn(Z_xx.shape[0], Z_xx.shape[1]) * np.complex(0, 1))

    # Inverse stft
    _, s_rec = signal.istft(Z_xx, fs=fs, window='boxcar', nperseg=ns, noverlap=no, boundary=None)

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


