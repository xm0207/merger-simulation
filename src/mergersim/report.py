"""Charts for the README and memo. Static PNGs drawn with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order: blue, orange, aqua
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_concentration(markets: pd.DataFrame, title: str, path: str | Path) -> None:
    """Post-merger HHI against delta HHI, one dot per market, with the EU guideline lines."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(markets["hhi_post"], markets["delta_hhi"], s=22, color=SERIES[0], edgecolor="white", lw=0.6,
               zorder=3)
    ymax = max(markets["delta_hhi"].max() * 1.15, 300)
    xmax = markets["hhi_post"].max() * 1.05
    ax.axvline(2000, color=MUTED, lw=1, ls="--")
    ax.plot([1000, 2000], [250, 250], color=MUTED, lw=1, ls=":")
    ax.plot([2000, xmax], [150, 150], color=MUTED, lw=1, ls=":")
    box = {"facecolor": "white", "edgecolor": "none", "pad": 1.5}
    ax.text(xmax * 0.99, 150 + ymax * 0.02, "delta = 150: EU HMG para. 20 safe harbour below this line\n(post-merger HHI above 2,000)",
            fontsize=7.5, color=MUTED, ha="right", va="bottom", bbox=box, zorder=4)
    ax.text(2000 + xmax * 0.008, ymax * 0.97, "HHI = 2,000", fontsize=7.5, color=MUTED, va="top", bbox=box)
    ax.set_xlim(min(1000, markets["hhi_post"].min() * 0.95), xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Post-merger HHI", color=MUTED, fontsize=9)
    ax.set_ylabel("Change in HHI (delta)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_price_effects(by_product: pd.DataFrame, rivals: pd.DataFrame, title: str, path: str | Path) -> None:
    """Average simulated price change of each merging product, and of all rivals, by demand model.

    `by_product` has columns model, product_ids, price_change_pct (merging products only);
    `rivals` has columns model, price_change_pct.
    """
    models = list(dict.fromkeys(by_product["model"]))
    products = sorted(by_product["product_ids"].unique())
    labels = products + ["Rivals (avg)"]
    y = np.arange(len(labels))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) * len(models) / 2 + 1.6))
    for i, (model, color) in enumerate(zip(models, SERIES)):
        vals = [float(by_product.query("model == @model and product_ids == @p")["price_change_pct"].iloc[0])
                for p in products]
        vals.append(float(rivals.query("model == @model")["price_change_pct"].iloc[0]))
        pos = y + (i - (len(models) - 1) / 2) * h
        ax.barh(pos, vals, height=h * 0.9, color=color, label=model, edgecolor="white", lw=1)
        for yy, v in zip(pos, vals):
            ax.text(v + 0.05, yy, f"{v:.1f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average price change after the merger, % (quantity-weighted over 94 markets)",
                  color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_efficiencies(table: pd.DataFrame, cmcr: dict, title: str, path: str | Path) -> None:
    """Merging-party price change against the assumed marginal-cost saving, one line per model.

    `table` has columns model, cost_reduction_pct, merging_price_change_pct; `cmcr` maps each
    model to its average compensating marginal-cost reduction (%), marked on its line.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (model, g), color in zip(table.groupby("model", sort=False), SERIES):
        ax.plot(g["cost_reduction_pct"], g["merging_price_change_pct"], color=color, lw=2, marker="o", ms=5,
                label=model)
        last = g.iloc[-1]
        ax.annotate(model, (last["cost_reduction_pct"], last["merging_price_change_pct"]), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center")
        if model in cmcr:
            ax.plot([cmcr[model]], [0], marker="|", ms=14, mew=2, color=color)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Assumed saving in the merging firms' marginal costs, %", color=MUTED, fontsize=9)
    ax.set_ylabel("Merging products' price change, %", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_xlim(right=table["cost_reduction_pct"].max() * 1.45)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
