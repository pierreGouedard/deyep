# Global import
import numpy as np
from scipy.sparse import save_npz, load_npz, vstack, hstack


# Local import
from deyep.utils.driver import driver


# Should inherit from FileDriver, that inherit from Driver
class NumpyDriver(driver.FileDriver):

    def __init__(self,):
        self.stream_orient, self.stream_n_cache, self.stream_is_sparse, self.stream_key, self.stream_url = \
            None, None, None, None, None

        driver.FileDriver.__init__(self, 'audio driver', 'Driver use to read / write any numpy arrays')

    def read_file(self, url, **kwargs):
        if kwargs.get('is_sparse', False):
            ax = load_npz(url)
        else:
            ax = np.load(url)

        return ax

    def read_partitioned_file(self, url, is_sparse=False, l_files=None):

        if l_files is None:
            l_files = self.listdir(url)

        d_out = {}
        for filename in l_files:
            if is_sparse:
                d_out[filename.split('.')[0]] = self.read_file(self.join(url, filename), **{'is_sparse': True})
            else:
                d_out[filename.split('.')[0]] = self.read_file(self.join(url, filename))

        return d_out

    def write_file(self, ax, url, **kwargs):
        if kwargs.get('is_sparse', False):
            save_npz(url, ax)
        else:
            np.save(url, ax)

    def write_partioned_file(self, d_ax, url, is_sparse=False):

        for k, v in d_ax.items():
            if is_sparse:
                self.write_file(v, self.join(url, '{}.{}'.format(k, '.npz')), **{'is_sparse': True})
            else:
                self.write_file(v, self.join(url, '{}.{}'.format(k, '.npy')))

    def init_stream_partionned_file(self, url, key_partition=lambda x: x, n_cache=2, orient=None, is_sparse=False,
                                    is_cyclic=False):

        # Initialize file and element cursor
        self.stream_orient, self.stream_n_cache, self.stream_is_sparse, self.stream_key, self.stream_url = \
            orient, n_cache, is_sparse, key_partition, url

        # Set current partition
        self.stream_step, self.stream_offset, self.stream_partitions = 0, 0, self.listdir(url)

        # Sort list of partitions
        self.stream_partitions = sorted(self.stream_partitions, key=key_partition)

        # TODO: put a function for next lines and handle the case we reached end of partition and the cyclic arguments
        # Cache files for stream
        self.stream_cache = self.read_partitioned_file(
            self.stream_url,
            l_files=self.stream_partitions[self.stream_offset: self.stream_offset + self.stream_n_cache],
            is_sparse=self.stream_is_sparse
        )

        self.stream_cache = self.gather_cache(self.stream_cache, self.stream_orient, self.stream_is_sparse,
                                              self.stream_key)

        self.stream_offset += self.stream_n_cache

        # Support at most matrices stream
        assert(self.stream_cache.ndim < 3, 'streaming of numpy array only available for dimension les than 3')

    def stream_next(self):

        assert(self.current_iteration is not None, 'Streaming has not been initiated')

        if self.stream_cache.ndim == 1:
            next_ = self.stream_cache[self.stream_step]

            if self.stream_step + 1 >= len(self.stream_cache):

                # TODO: put a function for next lines and handle the case we reached end of partition and the cyclic arguments
                # renew Cache files for stream
                self.stream_cache = self.read_partitioned_file(
                    self.stream_url,
                    l_files=self.stream_partitions[self.stream_offset: self.stream_offset + self.stream_n_cache],
                    is_sparse=self.stream_is_sparse
                )

                self.stream_cache = self.gather_cache(self.stream_cache, self.stream_orient, self.stream_is_sparse,
                                                      self.stream_key)
                self.current_iteration = 0
                self.stream_offset += self.stream_n_cache

            else:
                self.current_iteration += 1

        elif self.stream_orient == 'columns':

            next_ = self.stream_cache[:, self.stream_step]

            if self.stream_step + 1 >= self.stream_cache.shape[-1]:

                # TODO: put a function for next lines and handle the case we reached end of partition and the cyclic arguments
                # renew Cache files for stream
                self.stream_cache = self.read_partitioned_file(
                    self.stream_url,
                    l_files=self.stream_partitions[self.stream_offset: self.stream_offset + self.stream_n_cache],
                    is_sparse=self.stream_is_sparse
                )

                self.stream_cache = self.gather_cache(self.stream_cache, self.stream_orient,
                                                      self.stream_is_sparse,
                                                      self.stream_key)
                self.current_iteration = 0
                self.stream_offset += self.stream_n_cache

            else:
                self.current_iteration += 1
        else:

            next_ = self.stream_cache[self.stream_step, :]

            if self.stream_step + 1 >= self.stream_cache.shape[0]:

                # TODO: put a function for next lines and handle the case we reached end of partition and the cyclic arguments
                # renew Cache files for stream
                self.stream_cache = self.read_partitioned_file(
                    self.stream_url,
                    l_files=self.stream_partitions[self.stream_offset: self.stream_offset + self.stream_n_cache],
                    is_sparse=self.stream_is_sparse
                )

                self.stream_cache = self.gather_cache(self.stream_cache, self.stream_orient,
                                                      self.stream_is_sparse,
                                                      self.stream_key)
                self.current_iteration = 0
                self.stream_offset += self.stream_n_cache

            else:
                self.current_iteration += 1

        return next_

    @staticmethod
    def load_cache_stream():
        raise NotImplementedError

    @staticmethod
    def gather_cache(cache, orient, is_sparse, key):
        if is_sparse:
            if orient is None:
                cache = hstack([v for _, v in sorted(cache.items(),  key=lambda t: key(t[0]))])

            elif orient == 'columns':
                cache = hstack([v for _, v in sorted(cache.items(),  key=lambda t: key(t[0]))])

            else:
                cache = vstack([v for _, v in sorted(cache.items(), key = lambda t: key(t[0]))])

        else:
            if orient is None:
                cache = np.hstack((v for _, v in sorted(cache.items(), key=lambda t: key(t[0]))))

            elif orient == 'columns':
                cache = np.hstack((v for _, v in sorted(cache.items(),  key=lambda t: key(t[0]))))

            else:
                cache = np.vstack((v for _, v in sorted(cache.items(), key = lambda t: key(t[0]))))

        return cache

