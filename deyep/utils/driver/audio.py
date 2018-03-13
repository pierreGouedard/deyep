# Global import
from pydub import AudioSegment
from pydub.utils import mediainfo
import os
from numpy import array, hstack

# Local import
from deyep.utils.driver import driver


# Should inherit from FileDriver, that inherit from Driver
class AudioDriver(driver.FileDriver):

    allowed_format = ['.mp3', '.wav', '.flv', '.ogg']

    def __init__(self,):
        driver.FileDriver.__init__(self, 'audio driver', 'Driver use to read / write any sound file with extension '
                                                         '{}'.format(self.allowed_format))

    def get_format_from_extension(self, url):

        format = os.path.splitext(url)[1]

        if format not in self.allowed_format:
            raise TypeError("Format not understood {}".format(format))

        return format

    def read_file(self, url, **kwargs):

        # Get kwargs
        nb_channel = kwargs.get('nb_channel', None)
        sampling_rate = kwargs.get('sampling_rate', None)

        # Set format
        format = self.get_format_from_extension(url)

        # Get properties of audio file
        info = mediainfo(url)
        sampling_rate_ = int(info['sample_rate'])
        nb_channel_ = int(info['channels'])

        # read file
        audio = None

        if format == '.mp3':
            audio = AudioSegment.from_mp3(url)
        if format == '.wav':
            audio = AudioSegment.from_wav(url)
        if format == '.flv':
            audio = AudioSegment.from_flv(url)
        if format == '.ogg':
            audio = AudioSegment.from_ogg(url)

        if nb_channel is not None:
            if nb_channel_ != nb_channel:
                audio = audio.set_channels(nb_channel)
        else:
            nb_channel = nb_channel_

        if sampling_rate is not None:
            if sampling_rate_ != sampling_rate:
                raise NotImplementedError("Resampling of read audio file is not available yet")
        else:
            sampling_rate = sampling_rate_

        return audio, sampling_rate, nb_channel

    def read_array_from_file(self, url, nb_channel=None, sampling_rate=None, order_out='fortran'):

        audio, sampling_rate, nb_channel = self.read_file(url, nb_channel=nb_channel, sampling_rate=sampling_rate)

        if nb_channel > 1:
            if order_out == 'fortran':
                ax_audio = array(audio.get_array_of_samples(), dtype=int)

            elif order_out == 'scipy':
                l_audio = audio.split_to_mono()
                ax_audio = array(l_audio[0].get_array_of_samples(), dtype=int).transpose()

                for x in l_audio[1:]:
                    ax_audio = hstack((ax_audio, array(x.get_array_of_samples(), dtype=int).transpose()))

            else:
                raise ValueError('order_out not understood choose from {}'.format(['F', 'scipy']))
        else:
            ax_audio = array(audio.get_array_of_samples(), dtype=int)

        return ax_audio, sampling_rate, nb_channel

    def write_array_to_file(self, format):
        raise NotImplementedError

    def write_file(self, **kwargs):
        raise NotImplementedError










