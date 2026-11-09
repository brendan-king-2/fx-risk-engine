"""Static figures for the README (the interactive version lives in dashboard/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEG, POS = "#d03b3b", "#2a78d6"  # loss / gain (diverging poles)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.frameon": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.axisbelow": True,
})


def _money(x, _=None):
    a = abs(x)
    if a < 1:
        return "$0"
    s = f"${a/1e6:.1f}M" if a >= 1e6 else f"${a/1e3:.0f}K"
    return ("-" if x < 0 else "") + s


def backtest_chart(pnl, var_a, var_b, label_a, label_b, path: Path):
    df = pd.concat([pnl, var_a.rename("a"), var_b.rename("b")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.vlines(df.index, 0, df["pnl"], color=AXIS, linewidth=0.5, label="Daily P&L")
    ax.plot(df.index, -df["b"], color=ORANGE, linewidth=1.3, label=f"99% VaR: {label_b}")
    ax.plot(df.index, -df["a"], color=BLUE, linewidth=1.3, label=f"99% VaR: {label_a}")
    hit = df[-df["pnl"] > df["a"]]
    ax.scatter(hit.index, hit["pnl"], s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3,
               label=f"Exceptions ({label_a}): {len(hit)}")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(_money))
    ax.set_title("Daily P&L vs. one-day 99% VaR forecast, 2008-2026", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9, borderaxespad=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def exceptions_chart(table: pd.DataFrame, expected: float, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(table))
    ax.barh(y, table["exceptions"], height=0.5, color=BLUE)
    ax.axvline(expected, color=INK2, linewidth=1)
    ax.text(expected, 1.0, f" expected {expected:.0f}", color=INK2, fontsize=9, va="bottom",
            transform=ax.get_xaxis_transform())
    for yi, v in zip(y, table["exceptions"]):
        ax.text(v + 1, yi, str(int(v)), va="center", color=INK, fontsize=9)
    ax.set_yticks(y, table.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_title(f"99% VaR exceptions over {int(table['days'].iloc[0]):,} days", pad=18)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def rolling_exceptions_chart(roll: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    top = max(12, roll.max().max() + 2)
    ax.axhspan(-0.5, 4.5, color="#0ca30c", alpha=0.07, linewidth=0)
    ax.axhspan(4.5, 9.5, color="#fab219", alpha=0.10, linewidth=0)
    ax.axhspan(9.5, top, color="#d03b3b", alpha=0.07, linewidth=0)
    for name, color in zip(roll.columns, [BLUE, ORANGE]):
        ax.plot(roll.index, roll[name], color=color, linewidth=1.5, label=name)
    for y, lab in [(2, "Green\n(0-4)"), (7, "Yellow\n(5-9)"), (12, "Red\n(10+)")]:
        ax.text(1.01, y, lab, color=INK2, fontsize=8.5, va="center", transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.5, top); ax.set_title("Basel traffic light: 99% VaR exceptions in trailing 250 days")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def component_chart(table: pd.DataFrame, path: Path):
    t = table.sort_values("component_var")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [NEG if v > 0 else POS for v in t["component_var"]]
    ax.barh(t.index, t["component_var"], height=0.5, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    for i, v in enumerate(t["component_var"]):
        ax.text(v + (6e3 if v >= 0 else -6e3), i, _money(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    lo, hi = t["component_var"].min(), t["component_var"].max()
    ax.set_xlim(min(lo * 1.8, -0.15 * hi), hi * 1.15)
    ax.xaxis.set_major_formatter(mt.FuncFormatter(_money)); ax.grid(axis="y", visible=False)
    ax.set_title("Component 99% VaR by currency (red adds risk, blue offsets it)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def stress_chart(stress: pd.DataFrame, path: Path):
    s = stress.sort_values("pnl", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [NEG if v < 0 else POS for v in s["pnl"]]
    ax.barh(s["scenario"], s["pnl"], height=0.55, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    for i, v in enumerate(s["pnl"]):
        ax.text(v + (1.5e5 if v >= 0 else -1.5e5), i, _money(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    lo = s["pnl"].min() * 1.25
    ax.set_xlim(lo, max(s["pnl"].max() * 1.6, 1e6))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(_money)); ax.grid(axis="y", visible=False)
    ax.set_title("Stress tests: P&L on today's exposures")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
