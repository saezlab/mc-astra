"""Top-level namespace for the mc-ASTRA package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import down, pl, up

__all__ = ["pl", "up", "down"]

try:
    __version__ = version("mc-astra")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"
