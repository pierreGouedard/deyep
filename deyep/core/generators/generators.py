# Global imports
import pickle
import os

# Local imports
import settings


class Generators(object):

    def __init__(self, project, dirin, dirout):

        self.project = project
        self.dir_gen = settings.deyep_generator_path.format(project)
        self.dirin = dirin
        self.dirout = dirout
        self.raw_data = None
        self.raw_features = None

    def read_raw_data(self, url=None):
        raise NotImplementedError

    def read_raw_features(self, url):
        raise NotImplementedError

    def write_raw_features(self, url=None):
        raise NotImplementedError

    def run_preprocessing(self):
        raise NotImplementedError

    def run_postprocessing(self, d_raw_features):
        raise NotImplementedError

    def save(self):
        import IPython
        IPython.embed()
        with open(os.path.join(self.dir_gen, 'generator.pickle'), 'wb') as handle:
            pickle.dump(self, handle)

