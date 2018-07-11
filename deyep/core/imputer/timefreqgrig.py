# global import
import os
import scipy.sparse
import numpy as np
from deyep.utils.driver.nmp import NumpyDriver
from deyep.utils.driver.audio import AudioDriver

# Local import
from deyep.core.imputer.comon import ImputerSingleSource
from deyep.utils.signal_processing.sounds import compute_stft_decomposition, optimize_segmentation, \
    butter_lowpass_filter, inverse_stft_decomposition
from deyep.utils.signal_processing.various import Discretizer, Normalizer


class SingleTimeFreqGridGenerator(ImputerSingleSource):

    def __init__(self, project, dirin, dirout, window='boxcar', noverlap=0, nperseg=2210, maxdurationsegment=10,
                 segoverlap=0.5, maxfrequency=6000, nb_channel=1, n_discrete=100):

        ImputerSingleSource.__init__(self, project, dirin, dirout)

        # Get source filename for input raw data
        self.src = os.listdir(self.dirin)[0]
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

    def read_raw_data(self, name):

        # Set driver and url if necessary
        driver = AudioDriver()

        # read raw data
        self.raw_data, self.samplingrate, _ = driver.read_array_from_file(driver.join(self.dirin, name),
                                                                          **{'nb_channel': self.nb_channel})

        # Assert that attributes are ok
        assert self.maxdurationsegment * self.segoverlap * self.samplingrate > self.nperseg, \
            "The number of sample used for fft of each segment is higher than the length of overlapping windows for " \
            "decomposition"

    def read_features(self):

        # Set driver and urls
        driver = NumpyDriver()
        urlf, urlb = driver.join(self.dirout, self.name_forward), driver.join(self.dirout, self.name_backward)

        # Save input and output file as partitioner numpy array
        self.features_forward = driver.read_partitioned_file(urlf, is_sparse=True)
        self.features_backward = driver.read_partitioned_file(urlb, is_sparse=True)

    def write_features(self, name_forward, name_backward):

        # Set driver
        driver = NumpyDriver()

        # Save names
        self.name_forward, self.name_backward = name_forward, name_backwards

        # Remove raw features if previously built
        if driver.exists(driver.join(self.dirout, name_forward)):
            driver.remove(driver.join(self.dirout, name_forward), recursive=True)

        if driver.exists(driver.join(self.dirout, name_backward)):
            driver.remove(driver.join(self.dirout, name_backward), recursive=True)

        # Create  output directory
        driver.makedirs(driver.join(self.dirout, name_forward))
        driver.makedirs(driver.join(self.dirout, name_backward))

        # Save input and output file as partitionner numpy array
        driver.write_partioned_file(self.features_forward, driver.join(self.dirout, name_forward), is_sparse=True)
        driver.write_partioned_file(self.features_forward, driver.join(self.dirout, name_backward), is_sparse=True)

    def run_preprocessing(self):

        # Normalize signal
        self.raw_data = self.normalizer.set_transform(self.raw_data)

        # Optimize parameter of decomposition
        self.nperseg = optimize_segmentation(self.nperseg, self.maxdurationsegment, self.samplingrate)

        # Low pass signal to remove unecessary high frequency noise
        self.raw_data = butter_lowpass_filter(self.raw_data, self.maxfrequency, self.samplingrate, order=5)

        # Decompose the Signal in multiple stft segment
        d_stft = compute_stft_decomposition(self.raw_data, self.maxdurationsegment, self.samplingrate, self.segoverlap,
                                            self.maxfrequency, self.noverlap, self.nperseg)

        # Init bins Discretize
        self.discretizer.set_discretizer_bins(np.hstack((d['re'] for k, d in d_stft.items())).flatten('F'),
                                              method='treshold', **{'treshold': 1e-3})

        # Encode the stft and build Input / Output features
        self.features_forward = dict()
        for k in d_stft.keys():
            self.features_forward[k] = scipy.sparse.vstack(
                [self.discretizer.encode_2d_array(d_stft[k]['re'], sparse=True, orient='columns'),
                 self.discretizer.encode_2d_array(d_stft[k]['im'], sparse=True, orient='columns')]
            )

            # update meta for stft inversion
            self.meta_stft.update({k: {'window': d_stft[k]['window'], 'size': len(d_stft[k]['freq'])}})

        self.features_backward = self.features_backward.copy()

    def run_postprocessing(self, d_features):
        # Build spectograms from features
        d_stft = self.meta_stft.copy()
        for k in d_features.keys():
            # Decode signal
            ax = self.discretizer.decode_2d_array(d_features[k], sparse=True, orient='columns')

            # Fill real and imaginary part
            d_stft[k]['re'] = ax[:d_stft[k]['size'], :]
            d_stft[k]['im'] = ax[d_stft[k]['size']:, :]

        # Inverse spectograms
        raw_data_out = inverse_stft_decomposition(d_stft, self.samplingrate, self.noverlap, self.nperseg)

        return raw_data_out
