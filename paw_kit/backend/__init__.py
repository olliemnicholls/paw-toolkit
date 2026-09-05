"""Backend abstraction layer for PAW compilation and inference."""

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend

__all__ = ["AbstractPAWBackend", "MockPAWBackend"]
