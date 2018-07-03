# Global imports
import pickle
import os

# Local imports
import settings


class Imputer(object):

    def __init__(self, project, dirin, dirout):

        self.dir_gen = settings.deyep_generator_path.format(project)
        self.project = project
        self.dirin = dirin
        self.dirout = dirout

    def read_raw_data(self, **kwargs):
        raise NotImplementedError

    def read_raw_features(self, **kwargs):
        raise NotImplementedError

    def write_raw_features(self, **kwargs):
        raise NotImplementedError

    def run_preprocessing(self):
        raise NotImplementedError

    def run_postprocessing(self, d_features):
        raise NotImplementedError

    def save(self):
        with open(os.path.join(self.dir_gen, 'imputer.pickle'), 'wb') as handle:
            pickle.dump(self, handle)


class ImputerSingleSource(Imputer):
    def __init__(self, project, dirin, dirout):
        Imputer.__init__(self, project, dirin, dirout)

        self.raw_data = None
        self.raw_features = None

    def read_raw_data(self, url=None):
        raise NotImplementedError

    def read_raw_features(self, url=None):
        raise NotImplementedError

    def write_raw_features(self, url=None):
        raise NotImplementedError


class ImputerDoubleSource(Imputer):
    def __init__(self, project, dirin, dirout):
        Imputer.__init__(self, project, dirin, dirout)

        self.raw_data_in = None
        self.raw_features_in = None
        self.raw_data_out = None
        self.raw_features_out = None

    def read_raw_data(self, urlin=None, urlout=None):
        raise NotImplementedError

    def read_raw_features(self, urlin=None, urlout=None):
        raise NotImplementedError

    def write_raw_features(self, urlin=None, urlout=None):
        raise NotImplementedError




