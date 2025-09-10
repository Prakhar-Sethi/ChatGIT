"""
Backward compatibility wrapper for api.py

This file maintains backward compatibility by importing from the new package structure.
For new code, import directly from chatgit.api.app instead.
"""

from chatgit.api.app import app

__all__ = ["app"]
