"""Historical and hypothetical stress scenarios applied to today's exposures."""
from __future__ import annotations

import pandas as pd

# (label, start close, end close, note)
HISTORICAL_SCENARIOS = [
    ("Lehman collapse / GFC", "2008-09-12", "2008-10-27", "Dash for dollars; carry trades unwind"),
    ("SNB drops EUR/CHF floor", "2015-01-14", "2015-01-15", "CHF jumps ~20% intraday"),
    ("Brexit referendum", "2016-06-23", "2016-06-27", "GBP worst move in decades"),
    ("COVID dash for cash", "2020-03-09", "2020-03-19", "Global USD funding squeeze"),
    ("2022 dollar surge", "2022-01-03", "2022-09-27", "Aggressive Fed hiking cycle"),
    ("Yen carry-trade unwind", "2024-07-10", "2024-08-05", "JPY rallies as carry trades close"),
    ("April 2025 tariff shock", "2025-04-02", "2025-04-21", "Broad USD sell-off"),
]

HYPOTHETICAL_SCENARIOS = {
    "Broad USD rally +10%": {c: -0.10 for c in ["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "MXN", "CNY"]},
    "EM / risk-off shock": {"MXN": -0.15, "AUD": -0.08, "CAD": -0.05, "GBP": -0.04,
                            "EUR": -0.03, "CNY": -0.02, "JPY": 0.05, "CHF": 0.04},
    "Safe-haven spike (JPY, CHF +15%)": {"JPY": 0.15, "CHF": 0.15},
    "CNY devaluation -8%": {"CNY": -0.08, "AUD": -0.04, "MXN": -0.03},
}


def historical_stress(prices: pd.DataFrame, exposures: pd.Series) -> pd.DataFrame:
    rows = []
    for label, start, end, note in HISTORICAL_SCENARIOS:
        p0 = prices.loc[:start].iloc[-1]
        p1 = prices.loc[:end].iloc[-1]
        moves = p1 / p0 - 1
        pnl_by_ccy = exposures * moves[exposures.index]
        rows.append({"scenario": label, "type": "historical", "window": f"{start} to {end}",
                     "note": note, "pnl": pnl_by_ccy.sum(),
                     **{f"move_{c}": moves[c] for c in exposures.index},
                     **{f"pnl_{c}": pnl_by_ccy[c] for c in exposures.index}})
    return pd.DataFrame(rows)


def hypothetical_stress(exposures: pd.Series) -> pd.DataFrame:
    rows = []
    for label, shocks in HYPOTHETICAL_SCENARIOS.items():
        moves = pd.Series(shocks).reindex(exposures.index).fillna(0.0)
        pnl_by_ccy = exposures * moves
        rows.append({"scenario": label, "type": "hypothetical", "window": "instantaneous",
                     "note": "", "pnl": pnl_by_ccy.sum(),
                     **{f"move_{c}": moves[c] for c in exposures.index},
                     **{f"pnl_{c}": pnl_by_ccy[c] for c in exposures.index}})
    return pd.DataFrame(rows)


def worst_window(returns: pd.DataFrame, exposures: pd.Series, horizon: int = 10) -> dict:
    """Worst `horizon`-day loss today's book would have suffered at any point in history."""
    growth = (1 + returns[exposures.index]).rolling(horizon).apply(lambda r: r.prod(), raw=True) - 1
    pnl = (growth * exposures).sum(axis=1).dropna()
    end = pnl.idxmin()
    start = returns.index[returns.index.get_loc(end) - horizon + 1]
    return {"horizon_days": horizon, "pnl": float(pnl.min()),
            "start": str(start.date()), "end": str(end.date())}
