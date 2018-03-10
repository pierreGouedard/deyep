# Global imports

# Local imports
import settings


class Generators(object):

    def __init__(self, project, driver):

        self.project = project
        self.driver = driver
        self.url_in = settings.deyep_raw_path.format(project)
        self.url_out = settings.deyep_io_path.format(project)
        self.raw_data = None
        self.raw_features = None

    def read_raw_data(self):
        self.raw_data = self.driver.read_content(self.url_in)

    def run_preprocessing(self):
        raise NotImplementedError

    def run_postprocessing(self):
        raise NotImplementedError

    def save_raw_features(self):
        raise NotImplementedError
