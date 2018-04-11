# global import
import os
import scipy.sparse
import numpy as np

# Local import
from deyep.core.generators.generators import Generators
from deyep.utils.signal_processing.sounds import compute_stft_decomposition, optimize_segmentation, \
    butter_lowpass_filter, inverse_stft_decomposition
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
        self.raw_data, self.samplingrate, _ = self.driver_in.read_array_from_file(
            self.driver_in.join(self.dir_in, self.src), **{'nb_channel': self.nb_channel}
        )

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

    def run_postprocessing(self, url=None):

        if url is None:
            url = self.driver_out.join(self.dir_out, 'output')

        # Load output
        d_raw_features = self.driver_out.read_partitioned_file(url, is_sparse=True)

        # Build spectograms from features
        d_stft = self.meta_stft.copy()
        for k in d_raw_features.keys():
            # Decode signal
            ax = self.discretizer.decode_2d_array(d_raw_features[k], sparse=True, orient='columns')

            # Fill real and imaginary part
            d_stft[k]['re'] = ax[:d_stft[k]['size'], :]
            d_stft[k]['im'] = ax[d_stft[k]['size']:, :]

        # Inverse spectograms
        raw_data_out = inverse_stft_decomposition(d_stft, self.samplingrate, self.noverlap, self.nperseg)

        return raw_data_out

    def save_raw_features(self):

        # Remove raw features if previously built
        if self.driver_out.exists(self.driver_out.join(self.dir_out, 'input')):
            self.driver_out.remove(self.driver_out.join(self.dir_out, 'input'), recursive=True)

        if self.driver_out.exists(self.driver_out.join(self.dir_out, 'output')):
            self.driver_out.remove(self.driver_out.join(self.dir_out, 'output'), recursive=True)

        # Create  output directory
        self.driver_out.makedirs(self.driver_out.join(self.dir_out, 'input'), recursive=True)
        self.driver_out.makedirs(self.driver_out.join(self.dir_out, 'output'), recursive=True)

        # Save input and output file as partitionner numpy array
        self.driver_out.write_partioned_file(self.raw_features, self.driver_out.join(self.dir_out, 'input'),
                                             is_sparse=True)
        self.driver_out.write_partioned_file(self.raw_features, self.driver_out.join(self.dir_out, 'output'),
                                             is_sparse=True)
