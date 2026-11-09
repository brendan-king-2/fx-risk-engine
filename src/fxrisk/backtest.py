"""Statistical backtests for VaR and ES forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def exceptions(pnl: pd.Series, var: pd.Series) -> pd.Series:
    """1 on days where the realised loss exceeded the VaR forecast."""
    return (-pnl > var).astype(int)


def kupiec_pof(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test. H0: exception rate == p."""
    T, x = len(hits), int(hits.sum())
    phat = x / T
    ll0 = xlogy(T - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(T - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence test. H0: exceptions don't cluster."""
    prev, curr = hits[:-1], hits[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def quantile_loss(pnl: np.ndarray, var: np.ndarray, p: float) -> float:
    """Average tick (pinball) loss of the VaR quantile -VaR at level p. Lower is better."""
    q = -var
    hit = (pnl < q).astype(float)
    return float(np.mean((p - hit) * (pnl - q)))


def acerbi_szekely_z2(pnl: np.ndarray, var: np.ndarray, es: np.ndarray, p: float) -> float:
    """Acerbi & Szekely (2014) Z2 ES test statistic.

    Expected value 0 when ES is correctly specified; significantly negative
    values mean tail losses are worse than ES predicted. Their simulations put
    the 5% critical value near -0.70 across a range of distributions.
    """
    loss = -pnl
    hit = loss > var
    return float(1 - np.sum(loss * hit / es) / (len(pnl) * p))


def basel_zone(n_exceptions: int) -> str:
    """Basel traffic light for 250 days of 99% VaR."""
    if n_exceptions <= 4:
        return "green"
    if n_exceptions <= 9:
        return "yellow"
    return "red"


def evaluate(pnl: pd.Series, fc: pd.DataFrame, alpha: float) -> dict:
    """Full backtest summary for one model at one confidence level."""
    df = pd.concat([pnl, fc], axis=1).dropna()
    p = 1 - alpha
    hits = exceptions(df["pnl"], df["var"]).to_numpy()
    T, x = len(hits), int(hits.sum())
    lr_pof, p_pof = kupiec_pof(hits, p)
    lr_ind, p_ind = christoffersen_ind(hits)
    lr_cc = lr_pof + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, 2)
    out = {
        "days": T,
        "exceptions": x,
        "expected": round(T * p, 1),
        "rate": x / T,
        "kupiec_p": p_pof,
        "christoffersen_ind_p": p_ind,
        "cond_coverage_p": p_cc,
        "quantile_loss": quantile_loss(df["pnl"].to_numpy(), df["var"].to_numpy(), p),
        "z2": acerbi_szekely_z2(df["pnl"].to_numpy(), df["var"].to_numpy(), df["es"].to_numpy(), p),
        "avg_var": float(df["var"].mean()),
        "avg_es": float(df["es"].mean()),
    }
    if abs(alpha - 0.99) < 1e-9:
        roll = pd.Series(hits, index=df.index).rolling(250).sum().dropna()
        zones = roll.map(basel_zone)
        out["pct_days_green"] = float((zones == "green").mean())
        out["pct_days_yellow"] = float((zones == "yellow").mean())
        out["pct_days_red"] = float((zones == "red").mean())
        out["current_250d_exceptions"] = int(roll.iloc[-1])
        out["current_zone"] = zones.iloc[-1]
    return out
