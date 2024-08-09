# --- Imports ------------------------------------------------------------------

# Standard library
import logging
import sys


# --- Logger -------------------------------------------------------------------

def setup(name: str = "flexibleSubsetSelection", level: int = logging.NOTSET):
    """
    Sets up the logger for the package.
    """
    log = logging.getLogger(name)
    if not log.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(level)
        log.propagate = False
    return log