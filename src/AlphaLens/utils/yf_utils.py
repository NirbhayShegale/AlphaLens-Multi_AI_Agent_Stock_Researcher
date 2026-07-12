"""
yf_utils.py — yfinance rate-limit helpers.

yfinance enforces a soft rate-limit of ~2 000 requests/hour per IP.
Calling yf_delay() before every Ticker() / Search() call keeps the
project well within that limit when multiple sub-agents fire in parallel.
"""

import time
import random


def yf_delay(base: float = 0.5, jitter: float = 0.3) -> None:
    """
    Sleep for `base` seconds ± random jitter before a yfinance call.

    Parameters
    ----------
    base   : float — minimum sleep in seconds (default 0.5 s)
    jitter : float — max extra random seconds added on top (default 0.3 s)
    """
    time.sleep(base + random.uniform(0, jitter))
