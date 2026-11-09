"""One-day-ahead Value-at-Risk and Expected Shortfall forecasts.

Every function takes the daily portfolio P&L series (USD) and returns a
DataFrame indexed like the input with columns `var` and `es`: positive numbers
expressing a loss. The forecast stored on day t uses information up to and
including day t-1 only, so the series can be compared directly with realised
P&L on day t (no look-ahead).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats


# ----------------------------------------------------------------------------
# Distribution helpers
# ----------------------------------------------------------------------------
def normal_var_es(sigma, alpha: float):
    """VaR and ES of a zero-mean normal loss with std `sigma` at confidence alpha."""
    z = stats.norm.ppf(alpha)
    return z * sigma, sigma * stats.norm.pdf(z) / (1 - alpha)


def std_t_var_es(nu: float, alpha: float):
    """VaR and ES multipliers for a Student-t rescaled to unit variance."""
    q = stats.t.ppf(alpha, nu)
    scale = np.sqrt((nu - 2) / nu)
    es = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return q * scale, es * scale


def _frame(index, var, es) -> pd.DataFrame:
    return pd.DataFrame({"var": var, "es": es}, index=index)


# ----------------------------------------------------------------------------
# 1. Historical simulation
# ----------------------------------------------------------------------------
def historical(pnl: pd.Series, alpha: float, window: int = 500) -> pd.DataFrame:
    """Empirical quantile of the last `window` days of P&L."""
    x = pnl.to_numpy()
    windows = np.sort(sliding_window_view(x, window)[:-1], axis=1)  # ends at t-1
    k = int(np.floor(window * (1 - alpha)))  # number of tail observations
    k = max(k, 1)
    var = -windows[:, k - 1]
    es = -windows[:, :k].mean(axis=1)
    pad = np.full(window, np.nan)
    return _frame(pnl.index, np.r_[pad, var], np.r_[pad, es])


# ----------------------------------------------------------------------------
# 2. Parametric (variance-covariance), equally weighted window
# ----------------------------------------------------------------------------
def parametric_normal(pnl: pd.Series, alpha: float, window: int = 500) -> pd.DataFrame:
    """Normal VaR with volatility from an equally weighted rolling window.

    With fixed USD exposures, w' Sigma w equals the variance of portfolio P&L,
    so this is identical to the full variance-covariance calculation.
    Daily mean is assumed zero (standard for one-day horizons).
    """
    sigma = np.sqrt((pnl**2).rolling(window).mean().shift(1))
    var, es = normal_var_es(sigma, alpha)
    return _frame(pnl.index, var, es)


# ----------------------------------------------------------------------------
# 3. EWMA / RiskMetrics
# ----------------------------------------------------------------------------
def ewma_sigma(pnl: pd.Series, lam: float = 0.94, seed: int = 60) -> pd.Series:
    """One-day-ahead EWMA volatility forecast: s2[t] = lam*s2[t-1] + (1-lam)*x[t-1]^2."""
    x = pnl.to_numpy()
    s2 = np.empty_like(x)
    s2[: seed + 1] = np.nan
    s2[seed] = np.mean(x[:seed] ** 2)
    for t in range(seed + 1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(s2), index=pnl.index)


def ewma_normal(pnl: pd.Series, alpha: float, lam: float = 0.94) -> pd.DataFrame:
    sigma = ewma_sigma(pnl, lam)
    var, es = normal_var_es(sigma, alpha)
    return _frame(pnl.index, var, es)


# ----------------------------------------------------------------------------
# 4. Filtered historical simulation (EWMA-standardised residuals)
# ----------------------------------------------------------------------------
def filtered_historical(pnl: pd.Series, alpha: float, window: int = 500, lam: float = 0.94) -> pd.DataFrame:
    """Rescale past standardised shocks by today's volatility (Barone-Adesi et al.)."""
    sigma = ewma_sigma(pnl, lam)
    z = (pnl / sigma).to_numpy()
    s = sigma.to_numpy()
    var = np.full(len(z), np.nan)
    es = np.full(len(z), np.nan)
    k = max(int(np.floor(window * (1 - alpha))), 1)
    for t in range(len(z)):
        hist = z[max(0, t - window): t]
        hist = hist[~np.isnan(hist)]
        if len(hist) < window or np.isnan(s[t]):
            continue
        tail = np.sort(hist)[:k]
        var[t] = -s[t] * tail[-1]
        es[t] = -s[t] * tail.mean()
    return _frame(pnl.index, var, es)


# ----------------------------------------------------------------------------
# 5. GARCH(1,1) with Student-t innovations
# ----------------------------------------------------------------------------
def garch_t(
    pnl: pd.Series,
    alphas: tuple[float, ...],
    window: int = 1000,
    refit_every: int = 20,
) -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    """Rolling GARCH(1,1)-t forecasts.

    Parameters are re-estimated every `refit_every` days on the trailing
    `window` days; between refits the variance recursion is rolled forward with
    the last estimated parameters using realised P&L, so each forecast only
    uses data available the evening before.

    Returns ({alpha: DataFrame[var, es]}, parameter history).
    """
    from arch import arch_model

    scale = 1_000.0  # fit in $ thousands for numerical stability
    x = pnl.to_numpy() / scale
    n = len(x)
    sig2 = np.full(n, np.nan)
    nu_t = np.full(n, np.nan)
    params = []

    for start in range(window, n, refit_every):
        sample = x[start - window: start]
        am = arch_model(sample, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        om, a1, b1, nu = (res.params[k] for k in ("omega", "alpha[1]", "beta[1]", "nu"))
        params.append({"date": pnl.index[start], "omega": om, "alpha": a1, "beta": b1,
                       "nu": nu, "persistence": a1 + b1})
        prev_s2 = res.conditional_volatility[-1] ** 2
        prev_e = sample[-1]
        for t in range(start, min(start + refit_every, n)):
            s2 = om + a1 * prev_e**2 + b1 * prev_s2
            sig2[t], nu_t[t] = s2, nu
            prev_s2, prev_e = s2, x[t]

    sigma = np.sqrt(sig2) * scale
    out = {}
    for alpha in alphas:
        var = np.full(n, np.nan)
        es = np.full(n, np.nan)
        ok = ~np.isnan(nu_t)
        # nu changes only at refits, so compute multipliers per distinct value
        for nu in np.unique(nu_t[ok]):
            m = nu_t == nu
            qv, qe = std_t_var_es(nu, alpha)
            var[m], es[m] = qv * sigma[m], qe * sigma[m]
        out[alpha] = _frame(pnl.index, var, es)
    return out, pd.DataFrame(params).set_index("date")


MODEL_LABELS = {
    "hs": "Historical simulation",
    "normal": "Parametric normal",
    "ewma": "EWMA (RiskMetrics)",
    "fhs": "Filtered historical sim.",
    "garch_t": "GARCH(1,1)-t",
}


def run_all(pnl: pd.Series, alphas=(0.99, 0.975)):
    """Forecast VaR/ES with every model. Returns ({model: {alpha: df}}, garch params)."""
    garch, garch_params = garch_t(pnl, alphas)
    forecasts = {
        "hs": {a: historical(pnl, a) for a in alphas},
        "normal": {a: parametric_normal(pnl, a) for a in alphas},
        "ewma": {a: ewma_normal(pnl, a) for a in alphas},
        "fhs": {a: filtered_historical(pnl, a) for a in alphas},
        "garch_t": garch,
    }
    return forecasts, garch_params
