"""Load daily FX rates and build the portfolio P&L series.

Source: Federal Reserve H.10 noon buying rates (as republished by FRED and
mirrored in the open `datasets/exchange-rates` repository). Every series in
that file is quoted as units of foreign currency per 1 USD.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "fx_daily.csv"

# ISO code -> country label used in the source file
CURRENCIES = {
    "EUR": "Euro",
    "JPY": "Japan",
    "GBP": "United Kingdom",
    "CHF": "Switzerland",
    "CAD": "Canada",
    "AUD": "Australia",
    "MXN": "Mexico",
    "CNY": "China",
}


def download(path: Path = RAW_PATH) -> Path:
    """Fetch the latest daily file (overwrites the local copy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, path)
    return path


def load_usd_prices(start: str = "2003-01-01", path: Path = RAW_PATH) -> pd.DataFrame:
    """USD value of one unit of each foreign currency, one column per ISO code.

    Inverts the source quote (FCY per USD -> USD per FCY) so that a rise means
    the foreign currency appreciated against the dollar. US market holidays have
    no fix for any currency and are dropped; an isolated gap in a single series
    is carried forward one day at most.
    """
    if not path.exists():
        download(path)
    raw = pd.read_csv(path, parse_dates=["Date"])
    wide = raw.pivot_table(index="Date", columns="Country", values="Exchange rate")
    wide = wide[list(CURRENCIES.values())]
    wide.columns = list(CURRENCIES.keys())
    wide = wide.loc[start:].ffill(limit=1).dropna()
    return 1.0 / wide


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns of the USD value of each currency."""
    return prices.pct_change().dropna()
