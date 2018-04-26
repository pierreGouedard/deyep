# Global import
import IPython
import os

# Local import
import settings
from deyep.core.generators.timefreqgrig import SingleTimeFreqGridGenerator

# Instantiate every element needed for simulation
project = 'test'
dir_in = settings.deyep_raw_path.format(project)
dir_out = settings.deyep_io_path.format(project)

# Create generator for audio signals
#generator = SingleTimeFreqGridGenerator(project, dir_in, dir_out)

# Read raw data
#generator.read_raw_data()

# Compute time freq grid
#generator.run_preprocessing()

# Save I/O
#generator.write_raw_features()

# Save generator
#generator.save()

import os
import pickle
from deyep.utils.driver.audio import AudioDriver
driver = AudioDriver()

with open(os.path.join(settings.deyep_generator_path.format(project), 'generator.pickle'), 'wb') as handle:
    generator = pickle.load(handle)

# Load raw fetures
raw_features = generator.read_raw_features(os.path.join(dir_out, 'input'))

# Build back song
raw_data_out = generator.run_postprocessing(raw_features)

# Build network


# Create deep network


