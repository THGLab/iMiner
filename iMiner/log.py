import logging
from typing import Optional
import os


def init_logger(logname: Optional[os.PathLike] = None) -> logging.Logger:
    # logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # file
    if logname is not None:
        handler = logging.FileHandler(str(logname))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

LOGGER = init_logger("iMiner.log")