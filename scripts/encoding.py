# Global import
from pathlib import Path
import numpy as np
from PIL import Image

import sys

project_path = Path(__file__.split('scripts')[0])
sys.path.append(project_path.as_posix())

from deyep.opencv.utils import apply_threshold
from deyep.shapes.encoder import ImageEncoder


if __name__ == '__main__':
    """
    Usage:
    python deyep/scripts/encoding.py

    """
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    numbers_path = data_path / "numbers"

    l_filenames = ['2_normal.png', '2_flip.png', '2_rotated.png', '2_translated.png']

    # Parameters
    threshold = 100
    n_features = 20

    for filename in l_filenames:
        # Read image
        image = Image.open(numbers_path / filename)

        # Apply treshold to isolate shape
        masked_image = apply_threshold(np.asarray(image), threshold)

        # Now start encoding
        img_encoder = ImageEncoder([masked_image], n_features)
        img_encoder.encode_image()



