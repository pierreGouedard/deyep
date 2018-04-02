# Global imports
import numpy as np
import unittest
from scipy.fftpack import fft, fftfreq, fftshift

# Local import
from deyep.utils import interactive_plots as ip
from deyep.utils.signal_processing.sounds import get_part, butter_lowpass, butter_lowpass_filter, \
    compute_stft_decomposition, inverse_stft_decomposition, optimize_segmentation
from deyep.utils.signal_processing.various import Discretizer, Normalizer

__maintainer__ = 'Pierre Gouedard'


class TestSoundUtils(unittest.TestCase):

    def setUp(self):

        # Parameter of transformation
        self.samplingrate = 10000
        self.maxdurationsegment = 10
        self.segoverlap = 0.5
        self.maxfrequency = 2500

        # Build a realistic test signal
        self.n_freq = 40
        self.n_sec = 32
        self.frequencies = np.random.randint(5, self.maxfrequency, self.n_freq)

        times = np .linspace(0, self.n_sec*self.samplingrate, self.n_freq + 1)
        self.times = np.array([max(0, min(max(times), x + int(self.samplingrate * 0.1 * np.random.randn())))
                               for x in times])

        self.time_bins = [(self.times[i], self.times[i+1]) for i in range(len(self.times) - 1)]
        self.time_bins[0] = (0, self.time_bins[0][1])
        self.time_bins[-1] = (self.time_bins[-1][0], self.n_sec * self.samplingrate)
        self.s = np.zeros(self.n_sec * self.samplingrate)

        for freq, bound in zip(self.frequencies, self.time_bins):
            s_ = np.cos(2*np.pi * (float(freq) / self.samplingrate) * np.arange(bound[0], bound[1]))
            s_ = np.hstack((np.zeros(int(bound[0])), s_, np.zeros(int((self.n_sec * self.samplingrate) - bound[1]))))
            self.s += s_

        # Normalize signal
        self.s = Normalizer().set_transform(self.s)

        # Set discretization params
        self.N = 100

        # stft parameters
        self.nperseg = 2048
        self.noverlap = 0

    def test_decomposition(self):
        """
        Check decomposition of signal array-like into same size arrays with an overlapping argument

        python -m unittest tests.signal_processing.sounds.TestSoundUtils.test_decomposition

        """
        # Test signal decomposition at left border
        [s, mask] = get_part(0, self.s, self.maxdurationsegment, self.samplingrate, self.segoverlap)
        [s_, mask_] = get_part(0, self.s, self.maxdurationsegment, self.samplingrate, 0)

        assert all(s[mask] == s_)
        assert all(s[mask] == self.s[:self.samplingrate * self.maxdurationsegment])
        assert float(len(s[mask])) / self.samplingrate == self.maxdurationsegment
        assert all(mask_)
        assert len(s[~mask]) == (self.segoverlap * self.samplingrate * self.maxdurationsegment)

        # Test signal decomposition at right border
        r = int(np.ceil(float(len(self.s)) / (self.samplingrate * self.maxdurationsegment)))

        [s, mask] = get_part(r-1, self.s, self.maxdurationsegment, self.samplingrate, self.segoverlap)
        [s_, mask_] = get_part(r-1, self.s, self.maxdurationsegment, self.samplingrate, 0)

        assert all(s[mask] == s_)
        assert all(s_ == self.s[(r-1) * self.samplingrate * self.maxdurationsegment:])
        assert all(mask_)
        assert len(s[~mask]) == (self.segoverlap * self.samplingrate * self.maxdurationsegment)

        # Test signal decomposition in the middle right
        [s, mask] = get_part(2, self.s, self.maxdurationsegment, self.samplingrate, self.segoverlap)
        [s_, mask_] = get_part(2, self.s, self.maxdurationsegment, self.samplingrate, 0)

        N = self.samplingrate * self.maxdurationsegment
        assert all(s[mask] == s_)
        assert all(s_ == self.s[2 * N: 3 * N])
        assert all(mask_)

        # Test signal decomposition in the middle left
        [s, mask] = get_part(1, self.s, self.maxdurationsegment, self.samplingrate, self.segoverlap)
        [s_, mask_] = get_part(1, self.s, self.maxdurationsegment, self.samplingrate, 0)

        N = self.samplingrate * self.maxdurationsegment
        assert all(s[mask] == s_)
        assert all(s_ == self.s[N: 2 * N])
        assert all(mask_)
        assert len(s[~mask]) == 2 * (self.segoverlap * self.samplingrate * self.maxdurationsegment)

    def test_low_pass_filter(self):
        """
        Check consruction of stress period

        python -m unittest tests.signal_processing.sounds.TestSoundUtils.test_low_pass_filter

        """

        # Get frequency power above cut off frequency
        yf = fft(self.s)
        xf = fftfreq(len(self.s), 1. / self.samplingrate)
        energy = sum(np.abs(yf[xf > self.maxfrequency + 100]))

        # low pass filter signal
        s_ = butter_lowpass_filter(self.s, self.maxfrequency, self.samplingrate)

        # Get frequency power above the cut off frequency
        yf_ = fft(s_)
        xf_ = fftfreq(len(s_), 1. / self.samplingrate)
        energy_ = sum(np.abs(yf_[xf_ > self.maxfrequency + 100]))

        assert energy_/energy < 1e-2

    def test_signal_reconstruction(self):
        """
        Check reconsruction of a signal that has gone through a 'by part' stft

        python -m unittest tests.signal_processing.sounds.TestSoundUtils.test_signal_reconstruction

        """

        # pre process signal (low pass filter)
        s_ = butter_lowpass_filter(self.s, self.maxfrequency, self.samplingrate)

        # Optimize parameter
        nperseg = optimize_segmentation(self.nperseg, self.maxdurationsegment, self.samplingrate)

        # Decompose the signal as stft
        d_stft = compute_stft_decomposition(s_, self.maxdurationsegment, self.samplingrate, self.segoverlap,
                                            self.maxfrequency, self.noverlap, nperseg)

        # Recompose the signal with istft
        s_rec = inverse_stft_decomposition(d_stft, self.samplingrate, self.noverlap, nperseg, noise=1e-9)

        # assert reconstrution is ok
        assert np.abs(s_ -s_rec).sum() < 1e-1

    def test_spectograms_discretization(self):
        """
        Check reconsruction of a signal that has gone through a 'by part' stft

        python -m unittest tests.signal_processing.sounds.TestSoundUtils.test_spectograms_discretization

        """
        # Optimize parameter
        nperseg = optimize_segmentation(self.nperseg, self.maxdurationsegment, self.samplingrate)

        # Decompose the signal as stft
        d_stft = compute_stft_decomposition(self.s, self.maxdurationsegment, self.samplingrate, self.segoverlap,
                                            self.maxfrequency, self.noverlap, nperseg)

        discretizer = Discretizer(self.N)

        # Init bins
        discretizer.set_discretizer_bins(np.hstack((d_['re'] for k, d_ in d_stft.items())).flatten('F'),
                                         method='treshold', **{'treshold': 1e-3})

        # Discretize the stft
        for k in d_stft.keys():
            d_stft[k]['re'] = discretizer.discretize_array(d_stft[k]['re'])
            d_stft[k]['im'] = discretizer.discretize_array(d_stft[k]['im'])

        assert len(np.unique(np.hstack((d_['re'] for k, d_ in d_stft.items())).flatten('F'))) < self.N

        # Recompose the signal with istft
        s_rec = inverse_stft_decomposition(d_stft, self.samplingrate, self.noverlap, nperseg)

        # assert reconstruction is ok
        assert np.abs(self.s - s_rec).mean() < 1e-1





