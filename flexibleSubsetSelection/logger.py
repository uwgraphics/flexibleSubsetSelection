# --- Imports ------------------------------------------------------------------

# Standard library
import logging
import sys


# --- Logger -------------------------------------------------------------------

def setup(name: str = "flexibleSubsetSelection", 
          level: int = logging.NOTSET) -> logging.Logger:
    """
    Sets up logging for the package.

    Inputs:
        name: The name of the logger, defaults to package level name.
        level: The level to set the logger to from Python logging.
    
    Returns:
        log: The Python logger object to be used for logging in the package.
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