# global import
import os
from scipy import signal
import matplotlib.pyplot as plt

# Local import
from deyep.core.generators.generators import Generators


class SingleTimeFreqGridGenerator(Generators):

    def __init__(self, project, driver):
        Generators.__init__(self, project, driver)

        # Get source filename for input raw data
        self.src_forward = filter(lambda x: 'forward' in x, os.listdir(self.dir_in))[0]
        self.src_backward = filter(lambda x: 'backward' in x, os.listdir(self.dir_in))[0]
        self.sampling_rate = None
        self.window = ('tukey', 0.25)
        self.noverlap = 0
        self.nperseg = 1000

    def read_raw_data(self):
        self.raw_data['forward'], self.sampling_rate, _ = \
            self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src_forward))
        self.raw_data['backward'],  _, _ = \
            self.driver.read_array_from_file(self.driver.join(self.dir_in, self.src_backward),
                                             **{'sampling_rate': self.sampling_rate})

    def run_preprocessing(self):
        import IPython
        IPython.embed()

        f, t, Sxx = signal.spectrogram(self.raw_data['forward'], self.sampling_rate,
                                       noverlap=self.noverlap, nperseg=self.nperseg)
        f_ = f[:60]
        Sxx_ = Sxx[:60, :]
        fig = plt.Figure()
        plt.pcolormesh(t, f_, Sxx_)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def run_postprocessing(self):
        raise NotImplementedError

    def save_raw_features(self):
        raise NotImplementedError
