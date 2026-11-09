"""Decompose portfolio VaR by currency (Euler / component VaR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94, seed: int = 60) -> pd.DataFrame:
    """EWMA covariance matrix forecast for the day after the last observation."""
    r = returns.to_numpy()
    cov = np.cov(r[:seed].T, bias=True)
    for t in range(seed, len(r)):
        x = r[t][:, None]
        cov = lam * cov + (1 - lam) * (x @ x.T)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def component_var(cov: pd.DataFrame, exposures: pd.Series, alpha: float = 0.99) -> tuple[pd.DataFrame, dict]:
    """Euler allocation of normal VaR. Components sum exactly to portfolio VaR.

    marginal_i  = z * (Sigma w)_i / sigma_p      (VaR change per $1 more exposure)
    component_i = w_i * marginal_i
    standalone_i = z * |w_i| * sigma_i           (VaR if this were the only position)
    """
    w = exposures[cov.index].to_numpy(dtype=float)
    S = cov.to_numpy()
    z = stats.norm.ppf(alpha)
    sigma_p = float(np.sqrt(w @ S @ w))
    marginal = z * (S @ w) / sigma_p
    component = w * marginal
    standalone = z * np.abs(w) * np.sqrt(np.diag(S))
    vols = np.sqrt(np.diag(S)) * np.sqrt(252)
    table = pd.DataFrame({
        "exposure": w,
        "ann_vol": vols,
        "standalone_var": standalone,
        "marginal_var": marginal,
        "component_var": component,
        "pct_of_total": component / component.sum(),
    }, index=cov.index)
    port_var = z * sigma_p
    summary = {
        "portfolio_var": port_var,
        "sum_standalone_var": float(standalone.sum()),
        "diversification_benefit": float(standalone.sum() - port_var),
        "diversification_pct": float(1 - port_var / standalone.sum()),
    }
    corr = S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))
    summary["correlation"] = pd.DataFrame(corr, index=cov.index, columns=cov.index)
    return table, summary
