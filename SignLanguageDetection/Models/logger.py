import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Models Folder initialized")
    logger.info("Models for sign detections")
    return logger

def valuefun():
    val = 4
    return val