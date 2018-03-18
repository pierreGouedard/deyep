# global import
import os
from scipy import signal
import numpy as np

# Local import
from deyep.core.generators.generators import Generators
from deyep.utils.signal_processing.sounds import compute_stft_decomposition, adjust_segmentation


class SingleTimeFreqGridGenerator(Generators):

    def __init__(self, project, driver, window=('tukey', 0.25), noverlap=0, nperseg=2210,
                 maxdurationsegment=10, segoverlap=0.5, maxfrequency=6000):

        Generators.__init__(self, project, driver)

        # Get source filename for input raw data
        self.src_forward = filter(lambda x: 'forward' in x, os.listdir(self.dir_in))[0]
        self.src_backward = filter(lambda x: 'backward' in x, os.listdir(self.dir_in))[0]

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
        self.raw_data['forward'], self.samplingrate, _ = \
            self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src_forward))
        self.raw_data['backward'],  _, _ = \
            self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src_backward),
                                             **{'sampling_rate': self.samplingrate})

        # Assert that attributes are ok
        assert self.maxdurationsegment * self.segoverlap * self.samplingrate > self.nperseg, \
            "The number of sample used for fft of each segment is higher than the length of overlapping windows for " \
            "decomposition"

        # Optimize parameter of decomposition
        self.nperseg = adjust_segmentation(self.nperseg, self.maxdurationsegment, self.segoverlap, self.samplingrate)

    def run_preprocessing(self):
        # Low pass filter forward and backward message

        # Decompose forward and backward signal
        d_stft = compute_stft_decomposition(self.raw_data['forward'], self.maxdurationsegment, self.samplingrate,
                                            self.segoverlap, self.maxfrequency, self.noverlap,  self.nperseg)


    def run_postprocessing(self):
        raise NotImplementedError

    def save_raw_features(self):
        raise NotImplementedError
