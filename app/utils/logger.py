import os
import sys

from loguru import logger

from app.config import get_settings


def setup_logger() -> None:
    """Initialize application, error, and access logs."""

    settings = get_settings()
    os.makedirs(settings.log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        enqueue=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    )
    logger.add(
        os.path.join(settings.log_dir, "app.log"),
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        encoding="utf-8",
    )
    logger.add(
        os.path.join(settings.log_dir, "error.log"),
        level="ERROR",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
    )
    logger.add(
        os.path.join(settings.log_dir, "access.log"),
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        encoding="utf-8",
        filter=lambda record: record["extra"].get("access_log", False),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    )


__all__ = ["logger", "setup_logger"]
