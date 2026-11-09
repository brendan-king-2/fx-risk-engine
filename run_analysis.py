"""Run the full FX market-risk pipeline.

    python run_analysis.py              # use the cached data file
    python run_analysis.py --refresh    # download the latest rates first

Writes tables to outputs/, figures to outputs/figures/, and the data the
dashboard reads to dashboard/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fxrisk import attribution, backtest, data, figures, models, portfolio, stress  # noqa: E402

ALPHAS = (0.99, 0.975)
BACKTEST_START = "2008-01-01"
OUT, FIG, DASH = ROOT / "outputs", ROOT / "outputs" / "figures", ROOT / "dashboard"


def main(refresh: bool = False) -> dict:
    for d in (OUT, FIG, DASH):
        d.mkdir(parents=True, exist_ok=True)
    if refresh:
        data.download()

    # ---- data -------------------------------------------------------------
    prices = data.load_usd_prices()
    rets = data.daily_returns(prices)
    exposures = portfolio.EXPOSURES_USD
    pnl = portfolio.portfolio_pnl(rets, exposures)

    # Append a placeholder for the next business day so every model also
    # produces tomorrow's forecast (which only uses data through today).
    next_day = pnl.index[-1] + pd.offsets.BDay(1)
    pnl_ext = pd.concat([pnl, pd.Series([0.0], index=[next_day], name="pnl")])

    # ---- models -----------------------------------------------------------
    fc, garch_params = models.run_all(pnl_ext, ALPHAS)
    current = {m: {str(a): {"var": float(fc[m][a]["var"].iloc[-1]), "es": float(fc[m][a]["es"].iloc[-1])}
                   for a in ALPHAS} for m in fc}
    fc = {m: {a: df.iloc[:-1] for a, df in d.items()} for m, d in fc.items()}

    # ---- backtests --------------------------------------------------------
    bt_pnl = pnl.loc[BACKTEST_START:]
    bt = {str(a): {m: backtest.evaluate(bt_pnl, fc[m][a].loc[BACKTEST_START:], a) for m in fc}
          for a in ALPHAS}
    for a in ALPHAS:
        tbl = pd.DataFrame(bt[str(a)]).T
        tbl.index = [models.MODEL_LABELS[m] for m in tbl.index]
        tbl.to_csv(OUT / f"backtest_{int(a*1000)}.csv")

    hits99 = pd.DataFrame({m: backtest.exceptions(bt_pnl, fc[m][0.99]["var"].loc[BACKTEST_START:])
                           for m in fc})
    roll99 = hits99.rolling(250).sum().dropna()

    # ---- stress + attribution --------------------------------------------
    stress_df = pd.concat([stress.historical_stress(prices, exposures),
                           stress.hypothetical_stress(exposures)], ignore_index=True)
    stress_df.to_csv(OUT / "stress_tests.csv", index=False)
    worst10 = stress.worst_window(rets, exposures, 10)

    cov = attribution.ewma_covariance(rets[exposures.index])
    comp, comp_summary = attribution.component_var(cov, exposures, 0.99)
    comp.to_csv(OUT / "component_var.csv")
    corr = comp_summary.pop("correlation")
    garch_params.to_csv(OUT / "garch_parameters.csv")

    # ---- figures ----------------------------------------------------------
    lab = models.MODEL_LABELS
    figures.backtest_chart(bt_pnl, fc["fhs"][0.99]["var"], fc["normal"][0.99]["var"],
                           lab["fhs"], lab["normal"], FIG / "backtest_99.png")
    t99 = pd.DataFrame(bt["0.99"]).T
    t99.index = [lab[m] for m in t99.index]
    figures.exceptions_chart(t99, t99["expected"].iloc[0], FIG / "exceptions_99.png")
    figures.rolling_exceptions_chart(
        roll99[["fhs", "normal"]].rename(columns={"fhs": lab["fhs"], "normal": lab["normal"]}),
        FIG / "basel_traffic_light.png")
    figures.component_chart(comp, FIG / "component_var.png")
    figures.stress_chart(stress_df, FIG / "stress_tests.png")

    # ---- dashboard payload ------------------------------------------------
    def k(s):  # $ thousands, rounded, NaN -> None
        return [None if np.isnan(v) else round(float(v) / 1e3, 1) for v in s]

    ts = {
        "dates": [d.strftime("%Y-%m-%d") for d in bt_pnl.index],
        "pnl": k(bt_pnl),
        "var": {m: {str(a): k(fc[m][a]["var"].loc[BACKTEST_START:]) for a in ALPHAS} for m in fc},
        "es": {m: {str(a): k(fc[m][a]["es"].loc[BACKTEST_START:]) for a in ALPHAS} for m in fc},
    }
    results = {
        "meta": {
            "data_start": str(rets.index[0].date()), "data_end": str(rets.index[-1].date()),
            "backtest_start": str(bt_pnl.index[0].date()), "forecast_for": str(next_day.date()),
            "n_obs": len(rets), "gross_exposure": portfolio.gross_exposure(exposures),
            "net_exposure": float(exposures.sum()),
            "exposures": {c: float(v) for c, v in exposures.items()},
            "source": "Federal Reserve H.10 daily noon buying rates (via FRED / datasets/exchange-rates)",
        },
        "model_labels": lab,
        "current": current,
        "backtest": bt,
        "rolling_exceptions_99": {"dates": [d.strftime("%Y-%m-%d") for d in roll99.index],
                                  **{m: roll99[m].astype(int).tolist() for m in roll99}},
        "timeseries": ts,
        "stress": stress_df.replace({np.nan: None}).to_dict(orient="records"),
        "worst_10d": worst10,
        "attribution": {"table": comp.reset_index(names="ccy").to_dict(orient="records"),
                        **{k_: float(v) for k_, v in comp_summary.items()}},
        "correlation": {"labels": list(corr.index), "matrix": corr.round(3).values.tolist()},
        "garch": {"median_persistence": float(garch_params["persistence"].median()),
                  "median_nu": float(garch_params["nu"].median()), "n_refits": len(garch_params)},
        "ann_vol": {c: float(v) for c, v in (rets.loc[BACKTEST_START:].std() * np.sqrt(252)).items()},
        "pnl_stats": {"skew": float(bt_pnl.skew()), "excess_kurtosis": float(bt_pnl.kurt()),
                      "std": float(bt_pnl.std())},
    }
    (DASH / "results.json").write_text(json.dumps(results, default=float))
    return results


def _print_summary(r: dict) -> None:
    m = r["meta"]
    print(f"Data {m['data_start']} to {m['data_end']} ({m['n_obs']:,} days); "
          f"gross exposure ${m['gross_exposure']/1e6:.0f}M")
    print(f"\nOne-day forecast for {m['forecast_for']}:")
    for key, lab in r["model_labels"].items():
        c = r["current"][key]
        print(f"  {lab:<26} VaR99 ${c['0.99']['var']:>11,.0f}   ES97.5 ${c['0.975']['es']:>11,.0f}")
    print(f"\n99% backtest from {m['backtest_start']}:")
    for key, b in r["backtest"]["0.99"].items():
        print(f"  {r['model_labels'][key]:<26} {b['exceptions']:>3} exceptions "
              f"(exp {b['expected']:.0f})  Kupiec p={b['kupiec_p']:.3f}  "
              f"CC p={b['cond_coverage_p']:.3f}  red-zone days {b['pct_days_red']:.1%}")
    a = r["attribution"]
    print(f"\nDiversification benefit: ${a['diversification_benefit']:,.0f} "
          f"({a['diversification_pct']:.0%} of standalone VaR)")
    w = r["worst_10d"]
    print(f"Worst 10-day loss on today's book: ${w['pnl']:,.0f} ({w['start']} to {w['end']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download latest rates first")
    _print_summary(main(ap.parse_args().refresh))
