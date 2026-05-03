import sys

from loguru import logger

from config import LOG_DESTINATION, LOG_LEVEL, LOG_SERIALIZE


def setup_logging():
    logger.remove()

    # 1. Console handler
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # 2. File handler
    logger.add(
        LOG_DESTINATION,
        level=LOG_LEVEL,
        serialize=LOG_SERIALIZE,
        rotation="100 MB",
        retention="7 days",
        compression="zip",
    )

    return logger
