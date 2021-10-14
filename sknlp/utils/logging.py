import logging
import threading

_logger = None
_logger_lock = threading.Lock()


def get_logger():
    global _logger

    if _logger:
        return _logger
    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        logger = logging.getLogger("sknlp")
        logger.setLevel(logging.INFO)
        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not logging.getLogger().handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()
