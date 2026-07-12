"""Shared logging helper for the msutils package."""
import logging


def create_logger(name):
    """Return a console logger, attaching a handler only once per logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        cfmt = logging.Formatter("%(name)s - %(asctime)s %(levelname)s - %(message)s")
        log.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(cfmt)
        log.addHandler(console)
    return log
