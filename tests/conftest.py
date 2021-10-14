import os
import logging

from tensorflow import get_logger
from sknlp.utils.logging import logger

get_logger().setLevel(logging.ERROR)
logger.setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"