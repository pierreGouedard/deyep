# Global import
import IPython


# Local import
from deyep.utils.driver.audio import AudioDriver
from deyep.core.generators.timefreqgrig import SingleTimeFreqGridGenerator

# Instantiate every element needed for simulation
project = 'test'
driver = AudioDriver()
generator = SingleTimeFreqGridGenerator(project, driver)

# Read raw data
generator.read_raw_data()
IPython.embed()

# Compute time freq grid
generator.run_preprocessing()

# Select random features in

# Save forward and backward features



IPython.embed()