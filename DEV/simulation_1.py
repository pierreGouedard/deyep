# Global import
import IPython
import pickle

# Local import
import settings
from deyep.utils.driver.driver import FileDriver
# Local import
from deyep.utils.driver.audio import AudioDriver
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.generators.timefreqgrig import SingleTimeFreqGridGenerator

# Instantiate every element needed for simulation
project = 'test'
driver_in = AudioDriver()
driver_out = NumpyDriver()

# Create generator for audio signals
generator = SingleTimeFreqGridGenerator(project, driver_in, driver_out)

# Read raw data
generator.read_raw_data()

# Compute time freq grid
generator.run_preprocessing()

IPython.embed()

# Save I/O
generator.save_raw_features()

# Save generator
generator.save()

###### TEST SAVE GENERATOR
driver = FileDriver('filedriver', 'filedriver')
with open(driver.join(settings.deyep_generator_path, 'generator.pickle'), 'rb') as handle:
    generator_ = pickle.load(handle)

# Compare generator and generator_

###### TEST SAVE GENERATOR

#### TEST POST PROCESSING



#### TEST POST PROCESSING

# Build network


# Create deep network


