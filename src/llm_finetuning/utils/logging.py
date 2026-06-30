"""Logging setup using loguru."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru with console + optional file output."""
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    if log_file:
        logger.add(log_file, level=level, rotation="10 MB", retention="7 days")
