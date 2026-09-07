"""Compatibility entry point for hosts already configured with `main:app`.

Deploy this file together with the entire app/ package, not on its own.
"""

from app.main import app

__all__ = ["app"]
