# --- Imports ------------------------------------------------------------------

# Standard library
import logging
from pathlib import Path
import sys


# --- Logger -------------------------------------------------------------------


def setup(
    name: str = "flexibleSubsetSelection",
    level: int = logging.NOTSET,
    fileName: str | Path | None = None,
    formatter: logging.Formatter | None = None,
) -> logging.Logger:
    """
    Sets up logging for the package.

    Inputs:
        name: The name of the logger, defaults to package level name.
        level: The level to set the logger to from Python logging.
        fileName: Path to a file to save log to
        formatter: A custom Python logging Formatter to format the log

    Returns:
        log: A configured Python logger instance.
    """
    log = logging.getLogger(name)
    if not log.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatString = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
        formatter = formatter or logging.Formatter(formatString)
        handler.setFormatter(formatter)
        log.addHandler(handler)

        if fileName:
            fileName = Path(fileName)
            fileHandler = logging.FileHandler(fileName)
            fileHandler.setLevel(level)
            fileHandler.setFormatter(formatter)
            log.addHandler(fileHandler)

    log.setLevel(level)
    log.propagate = False
    log.debug("Logger %s initialized with level %s.", name, logging.getLevelName(level))

    return log
