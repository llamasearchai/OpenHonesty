import logging
import sys
from typing import Optional


def setup_logging(verbose: bool = False, log_level: Optional[int] = None) -> logging.Logger:
    """Setup logging configuration."""
    if log_level is None:
        log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Default logger
logger = get_logger(__name__)