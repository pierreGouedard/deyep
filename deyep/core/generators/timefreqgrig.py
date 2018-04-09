# global import
import os
import scipy.sparse
import numpy as np

# Local import
from deyep.core.generators.generators import Generators
from deyep.utils.signal_processing.sounds import compute_stft_decomposition, optimize_segmentation, butter_lowpass_filter
from deyep.utils.signal_processing.various import Discretizer, Normalizer


class SingleTimeFreqGridGenerator(Generators):

    def __init__(self, project, driver_in, driver_out, window='boxcar', noverlap=0, nperseg=2210, maxdurationsegment=10,
                 segoverlap=0.5, maxfrequency=6000, nb_channel=1, n_discrete=100):

        Generators.__init__(self, project, driver_in, driver_out)

        # Get source filename for input raw data
        self.src = os.listdir(self.dir_in)[0]
        self.nb_channel = nb_channel

        # Parameter of transformation
        self.samplingrate = None
        self.window = window
        self.noverlap = noverlap

        # parameter for decomposition of the segment (nperseg is optimized for usual music sampling: 44200 Hz)
        self.nperseg = nperseg
        self.maxdurationsegment = maxdurationsegment
        self.segoverlap = segoverlap
        self.maxfrequency = maxfrequency

        # Meta for stft inversion
        self.meta_stft = {}

        # Init discretizer & normalizer
        self.discretizer = Discretizer(n_discrete, method='bins')
        self.normalizer = Normalizer()

    def read_raw_data(self):
        # read raw data
        self.raw_data, self.samplingrate, _ = self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src),
                                                                               **{'nb_channel': self.nb_channel})

        # Assert that attributes are ok
        assert self.maxdurationsegment * self.segoverlap * self.samplingrate > self.nperseg, \
            "The number of sample used for fft of each segment is higher than the length of overlapping windows for " \
            "decomposition"

    def run_preprocessing(self):

        # Normalize signal
        self.raw_data = self.normalizer.set_transform(self.raw_data)

        # Optimize parameter of decomposition
        self.nperseg = optimize_segmentation(self.nperseg, self.maxdurationsegment, self.samplingrate)

        # Low pass signal to remove unecessary  high frequency noise
        self.raw_data = butter_lowpass_filter(self.raw_data, self.maxfrequency, self.samplingrate, order=5)

        # Decompose the Signal in multiple stft segment
        d_stft = compute_stft_decomposition(self.raw_data, self.maxdurationsegment, self.samplingrate, self.segoverlap,
                                            self.maxfrequency, self.noverlap, self.nperseg)

        # Init bins Discretize
        self.discretizer.set_discretizer_bins(np.hstack((d['re'] for k, d in d_stft.items())).flatten('F'),
                                              method='treshold', **{'treshold': 1e-3})

        # Encode the stft and build Input / Output features
        for k in d_stft.keys():
            self.raw_features[k] = scipy.sparse.vstack(
                [self.discretizer.encode_2d_array(d_stft[k].pop('re'), sparse=True, orient='columns'),
                 self.discretizer.encode_2d_array(d_stft[k].pop('im'), sparse=True, orient='columns')]
            )

            # update meta for stft inversion
            self.meta_stft.update({k: {'window': d_stft[k].pop('window'), 'size': len(d_stft[k].pop('freq'))}})

    def run_postprocessing(self):
        # Load output

        # Build spectograms from output

        # post process spectograms ( join higher frequencies to respect COLA constraints)

        # Inverse spectograms

        raise NotImplementedError

    def save_raw_features(self):

        raise NotImplementedError
