import logging
import logging.handlers
import time
from pathlib import Path


def get_logger(name: str, level: str, log_dir_path: Path):
    logger = logging.getLogger(name)

    if level.lower() == 'debug':
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif level.lower() == 'info':
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
    elif level.lower() == 'warning':
        logging.basicConfig(level=logging.WARNING)
        logger.setLevel(logging.WARNING)
    elif level.lower() == 'error':
        logging.basicConfig(level=logging.ERROR)
        logger.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
    )

    file_handler = logging.handlers.WatchedFileHandler(
        log_dir_path / f"{time.strftime('%d-%m-%Y')}.log"
    )
    file_handler.setFormatter(formatter)

    return logger