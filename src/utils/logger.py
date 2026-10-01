import logging
import sys
import os
from typing import Optional

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class FlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for real-time logging."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(name: str = "arrhythmia_benchmark", log_file: Optional[str] = None) -> logging.Logger:
    """Configures and returns a clean, formatted, real-time flushed logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = FlushStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
