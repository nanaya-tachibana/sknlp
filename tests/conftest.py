import os
import logging
from absl import logging as absl_logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("sknlp").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl_logging.set_verbosity(logging.ERROR)