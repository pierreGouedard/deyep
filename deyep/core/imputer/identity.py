# global import
import os
import scipy.sparse
import numpy as np
from deyep.utils.driver.nmp import NumpyDriver
from deyep.utils.driver.audio import AudioDriver

# Local import
from deyep.core.imputer.comon import ImputerDoubleSource
from deyep.utils.signal_processing.sounds import compute_stft_decomposition, optimize_segmentation, \
    butter_lowpass_filter, inverse_stft_decomposition
from deyep.utils.signal_processing.various import Discretizer, Normalizer


class SingleTimeFreqGridGenerator(ImputerDoubleSource):

    def __init__(self, project, dirin, dirout, driver):

        ImputerDoubleSource.__init__(self, project, dirin, dirout)

        # Get source filename for input raw data
        self.srcin, self.srcout = {x for x in os.listdir(self.dirin) if 'in' in x}.pop(), \
                                  {x for x in os.listdir(self.dirin) if 'in' in x}.pop()

        self.driver = driver

    def read_raw_data(self, urlin=None, urlout=None):

        # Set driver and url if necessary
        driver = AudioDriver()
        if urlin is None:
            urlin = driver.join(self.dirin, self.srcin)

        if urlout is None:
            urlout = driver.join(self.dirin, self.srcout)

        # read raw data
        self.raw_data_in = self.driver.read_file(urlin)
        self.raw_data_out = self.driver.read_file(urlout)

    def read_raw_features(self, urlin=None, urlout=None):
        raise NotImplementedError

    def write_raw_features(self, urlin=None, urlout=None):
        raise NotImplementedError

    def run_preprocessing(self):
        self.raw_features_in = self.raw_data_in.copy()
        self.raw_features_out = self.raw_data_out.copy()

    def run_postprocessing(self, d_features):
        raise NotImplementedError