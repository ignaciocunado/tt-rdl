import logging
import os
import sys
from datetime import datetime


def logger_setup(log_dir: str = "logs"):
    """Sets up logging for the project.
    Args:
        log_dir: the directory to save logs to. Defaults to "logs".
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(os.path.join(log_dir, f"run_{timestamp}.log"))),  ## log to local log file
            logging.StreamHandler(sys.stdout),  ## log also to stdout (i.e., print to screen)
        ],
    )
