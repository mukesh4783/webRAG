"""WebRAG — grounded, change-aware Q&A over live web pages."""

from .config import Settings
from .engine import Engine

__all__ = ["Engine", "Settings"]
__version__ = "2.0.0"
