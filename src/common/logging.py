import logging
import os


def get_logger(name: str):
    level = os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(funcName)s | %(message)s",
    )

    return logging.getLogger(name)
