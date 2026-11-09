"""Portfolio definition.

A hypothetical US-headquartered company's net foreign-currency exposure, in USD.
Positive = long the currency (foreign revenue, receivables, overseas assets);
negative = short (foreign-currency debt or supplier payables). Exposures are held
at constant USD notional, so daily P&L is linear in currency returns.
"""
from __future__ import annotations

import pandas as pd

EXPOSURES_USD = pd.Series(
    {
        "EUR": 30_000_000,   # European subsidiary revenue
        "JPY": 15_000_000,   # Japanese distribution business
        "GBP": 15_000_000,   # UK subsidiary
        "CAD": 10_000_000,   # Canadian sales
        "AUD": 8_000_000,    # Australian sales
        "MXN": 12_000_000,   # Mexican manufacturing receivables
        "CHF": -10_000_000,  # CHF-denominated loan
        "CNY": -15_000_000,  # payables to Chinese suppliers
    },
    name="exposure_usd",
)


def gross_exposure(exposures: pd.Series = EXPOSURES_USD) -> float:
    return float(exposures.abs().sum())


def portfolio_pnl(returns: pd.DataFrame, exposures: pd.Series = EXPOSURES_USD) -> pd.Series:
    """Daily P&L in USD: sum of exposure_i * return_i."""
    return (returns[exposures.index] * exposures).sum(axis=1).rename("pnl")
