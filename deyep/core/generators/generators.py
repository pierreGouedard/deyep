# Global imports

# Local imports
import settings


class Generators(object):

    def __init__(self, project, driver):

        self.project = project
        self.driver = driver
        self.dir_in = settings.deyep_raw_path.format(project)
        self.dir_out = settings.deyep_io_path.format(project)
        self.raw_data = None
        self.raw_features = None

    def read_raw_data(self):
        raise NotImplementedError

    def run_preprocessing(self):
        raise NotImplementedError

    def run_postprocessing(self):
        raise NotImplementedError

    def save_raw_features(self):
        raise NotImplementedError
