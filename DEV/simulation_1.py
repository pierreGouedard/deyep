# Global import
import IPython


# Local import
from deyep.utils.driver.audio import AudioDriver
from deyep.utils.driver.numpy import NumpyDriver
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

# Save IO
generator.save_raw_features()

# Build network


# Create deep network


