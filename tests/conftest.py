"""
tests/conftest.py — pytest configuration for LUMINE.

Ensures the project root is on sys.path so all `app.*` imports work
without installing the package.
"""
import sys
import os

# Project root is one level above this file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
