"""Correctness checks for the risk engine. Run with `pytest -q`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxrisk import attribution, backtest, models  # noqa: E402

RNG = np.random.default_rng(7)


def test_normal_var_es_closed_form():
    var, es = models.normal_var_es(1.0, 0.99)
    assert var == pytest.approx(2.3263, abs=1e-4)
    assert es == pytest.approx(2.6652, abs=1e-4)


def test_student_t_es_matches_monte_carlo():
    nu, alpha = 5.0, 0.975
    var, es = models.std_t_var_es(nu, alpha)
    draws = stats.t.rvs(nu, size=2_000_000, random_state=1) * np.sqrt((nu - 2) / nu)
    losses = -draws
    q = np.quantile(losses, alpha)
    assert var == pytest.approx(q, rel=0.01)
    assert es == pytest.approx(losses[losses >= q].mean(), rel=0.01)


def test_forecasts_have_no_lookahead():
    """Changing today's P&L must not change today's forecast."""
    pnl = pd.Series(RNG.normal(size=800), index=pd.bdate_range("2020-01-01", periods=800))
    shocked = pnl.copy()
    shocked.iloc[-1] = -50.0
    for fn in (models.historical, models.parametric_normal, models.ewma_normal, models.filtered_historical):
        a, b = fn(pnl, 0.99), fn(shocked, 0.99)
        assert a.iloc[-1]["var"] == pytest.approx(b.iloc[-1]["var"])


def test_correct_model_passes_kupiec():
    """Normal data + normal VaR at the true sigma should give ~1% exceptions."""
    x = RNG.normal(size=20_000)
    hits = (-x > stats.norm.ppf(0.99)).astype(int)
    _, p = backtest.kupiec_pof(hits, 0.01)
    assert p > 0.01


def test_kupiec_rejects_too_many_exceptions():
    hits = np.zeros(1000, dtype=int)
    hits[:30] = 1  # 3% vs 1% expected
    _, p = backtest.kupiec_pof(hits, 0.01)
    assert p < 0.001


def test_christoffersen_detects_clustering():
    hits = np.zeros(1000, dtype=int)
    hits[100:110] = 1  # ten exceptions in a row
    _, p = backtest.christoffersen_ind(hits)
    assert p < 0.001


def test_component_var_sums_to_portfolio_var():
    r = pd.DataFrame(RNG.multivariate_normal([0, 0, 0], [[1, .5, .2], [.5, 1, .3], [.2, .3, 1]], 500) * 0.01,
                     columns=list("ABC"))
    cov = attribution.ewma_covariance(r)
    table, summary = attribution.component_var(cov, pd.Series({"A": 10.0, "B": -4.0, "C": 6.0}))
    assert table["component_var"].sum() == pytest.approx(summary["portfolio_var"])
    assert summary["diversification_benefit"] > 0


def test_basel_zones():
    assert [backtest.basel_zone(n) for n in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]
