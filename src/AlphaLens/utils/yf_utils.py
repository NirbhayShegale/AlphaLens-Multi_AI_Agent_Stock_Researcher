"""
Utility helpers for yfinance requests.

Adds a random 1–3 second sleep before every yfinance network call to
avoid hitting Yahoo Finance's rate-limit / 429 errors.
"""

import time
import random


def yf_delay() -> None:
    """Sleep for a random duration between 1 and 3 seconds."""
    delay = random.uniform(1, 3)
    time.sleep(delay)
