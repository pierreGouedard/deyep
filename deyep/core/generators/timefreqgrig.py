# global import
import os
from scipy import signal
import numpy as np

# Local import
from deyep.core.generators.generators import Generators
from deyep.utils.signal_processing.sounds import compute_stft_decomposition, optimize_segmentation, butter_lowpass_filter


class SingleTimeFreqGridGenerator(Generators):

    def __init__(self, project, driver, window='boxcar', noverlap=0, nperseg=2210,
                 maxdurationsegment=10, segoverlap=0.5, maxfrequency=6000, nb_channel=1):

        Generators.__init__(self, project, driver)

        # Get source filename for input raw data
        self.src_forward = filter(lambda x: 'forward' in x, os.listdir(self.dir_in))[0]
        self.src_backward = filter(lambda x: 'backward' in x, os.listdir(self.dir_in))[0]
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

    def read_raw_data(self):
        import IPython
        IPython.embed()

        self.raw_data['forward'], self.samplingrate, _ = \
            self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src_forward),
                                             **{'nb_channel': self.nb_channel})
        self.raw_data['backward'],  _, _ = \
            self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src_backward),
                                             **{'sampling_rate': self.samplingrate, 'nb_channel':self.nb_channel})

        # Assert that attributes are ok
        assert self.maxdurationsegment * self.segoverlap * self.samplingrate > self.nperseg, \
            "The number of sample used for fft of each segment is higher than the length of overlapping windows for " \
            "decomposition"

    def run_preprocessing(self):

        # Optimize parameter of decomposition
        self.nperseg = optimize_segmentation(self.nperseg, self.maxdurationsegment, self.samplingrate)

        # Low pass signal to remove unecessary  high frequency noise
        self.raw_data['forward'] = butter_lowpass_filter(self.raw_data['forward'], self.maxfrequency, self.samplingrate)

        # Decompose the Signal in multiple stft segment
        d_stft_forward = compute_stft_decomposition(self.raw_data['forward'], self.maxdurationsegment,
                                                    self.samplingrate, self.segoverlap, self.maxfrequency,
                                                    self.noverlap, self.nperseg)
        # Decompose forward and backward signal
        d_stft_backward = compute_stft_decomposition(self.raw_data['backward'], self.maxdurationsegment,
                                                     self.samplingrate, self.segoverlap, self.maxfrequency,
                                                     self.noverlap,  self.nperseg)

        # Discretize spectograms



        # Save forward and backward sources

    def run_postprocessing(self):
        # Load output

        # post process spectograms ( join higher frequencies to respect COLA constraints)

        # Inverse spectograms

        raise NotImplementedError

    def save_raw_features(self):
        raise NotImplementedError
