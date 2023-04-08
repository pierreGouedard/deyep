# Global import
import os
import sys
import random
from pathlib import Path
from PIL import Image

# Local import
project_path = Path(__file__.split('scripts')[0])
sys.path.append(project_path.as_posix())

from deyep.opencv.segment import OpenCVSeq
from deyep.ops.logger import get_logger


if __name__ == '__main__':
    """
    Usage:
    python scripts/opencv_seg.py
     
    """
    # Set path
    data_path = project_path / "data"
    log_dir_path = project_path / "logs"

    # Parameters
    image_name = "n01514668_10004.JPEG"
    seg_coef = 1.25
    n_iter = 5
    color_encoding = 'HSV'

    # get logger
    logger = get_logger('opencv_seg', "INFO", log_dir_path)

    # Load an image at random
    if image_name is None:
        l_images = list(os.listdir(data_path))
        image_name = random.choice(l_images)

    image = Image.open(Path(__file__).parent.parent / "data" / image_name)
    print(f"Segmenting {image_name}")

    opencv_seg = OpenCVSeq(image, seg_coef, n_iter, color_encoding, logger=logger)
    opencv_seg.segment()
    import IPython
    IPython.embed()
    # visualize image
