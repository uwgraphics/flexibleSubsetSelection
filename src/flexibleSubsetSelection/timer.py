# Based on https://realpython.com/python-timer/
# --- Imports ------------------------------------------------------------------

# Standard library
from contextlib import ContextDecorator
import time


# --- Timer --------------------------------------------------------------------
class TimerError(Exception):
    """A custom exception used to report errors in Timer usage."""

    pass


class Timer(ContextDecorator):
    def __init__(self):
        self._startTime = None

    def start(self):
        """Start a new timer."""
        if self._startTime is not None:
            raise TimerError("Timer is running. Use .stop() to stop it.")

        self._startTime = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer, and return the elapsed time."""
        if self._startTime is None:
            raise TimerError("Timer is not running. Use .start() to start it.")

        self.elapsedTime = time.perf_counter() - self._startTime
        self._startTime = None

    def __enter__(self):
        """Start a new timer as a context manager."""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer."""
        self.stop()
