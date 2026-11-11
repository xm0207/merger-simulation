"""Screen the hypothetical merger of Firm 3 and Firm 4 in Nevo's fake cereal data.

Everything here uses only shares, prices and stated assumptions: no demand is estimated.
Shares and HHIs are computed in SQL (DuckDB). Firm 1 + Firm 2 is screened as well, as a point
of comparison.

Writes to results/: firm_shares.csv, concentration_by_market.csv, concentration_summary.csv,
eu_hmg_classification.csv, critical_loss.csv, guppi_screen.csv and concentration.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mergersim import ShareTable, critical_elasticity, critical_loss, eu_hmg_indication, hhi
from mergersim.data import load_cereal
from mergersim.report import plot_concentration
from mergersim.screening import partner_mask, share_diversion
from mergersim.specs import CONTRAST_MERGER, MAIN_MERGER

STATED_MARGINS = (0.20, 0.35, 0.50)
STATED_OUTSIDE_DIVERSION = (0.0, 0.25, 0.50)


def guppi_screen(products: pd.DataFrame, merger, margin: float, outside: float) -> pd.DataFrame:
    """Firm-level GUPPIs with share-proportional diversion and one stated margin for every product.

    In each market, product j's diversion to product k is (1 - outside) * s_k / (S - s_j), with S
    the total share of the 24 brands. Product GUPPIs are revenue-weighted within each firm, then
    averaged over markets with equal weights.
    """
    rows = []
    for market_id, g in products.groupby("market_ids", sort=False):
        firms = g["firm_ids"].to_numpy()
        prices, shares = g["prices"].to_numpy(), g["shares"].to_numpy()
        D = share_diversion(shares, outside)
        mask = partner_mask(firms, merger.firms)
        product_guppi = (D * mask) @ (margin * prices) / prices
        to_partner = (D * mask).sum(axis=1)
        revenue = prices * shares
        for firm in merger.firms:
            f = firms == firm
            rows.append({"market_ids": market_id, "firm_ids": firm,
                         "diversion_to_partner": np.average(to_partner[f], weights=revenue[f]),
                         "guppi": np.average(product_guppi[f], weights=revenue[f])})
    out = pd.DataFrame(rows).groupby("firm_ids")[["diversion_to_partner", "guppi"]].mean().reset_index()
    return out.assign(stated_margin_pct=100 * margin, outside_diversion_pct=100 * outside,
                      diversion_to_partner_pct=lambda d: 100 * d["diversion_to_partner"],
                      guppi_pct=lambda d: 100 * d["guppi"]).drop(columns=["diversion_to_partner", "guppi"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    products = load_cereal()
    table = ShareTable(products)

    # Firm shares by volume (servings) and by value, averaged over the 94 city-quarter markets
    vol = table.average_firm_shares("volume").add_prefix("volume_").rename(columns={"volume_firm_ids": "firm_ids"})
    val = table.average_firm_shares("value")[["firm_ids", "mean_share_pct"]].rename(
        columns={"mean_share_pct": "value_mean_share_pct"})
    brands = products.groupby("firm_ids")["product_ids"].nunique().rename("brands").reset_index()
    shares = brands.merge(vol.drop(columns="volume_markets"), on="firm_ids").merge(val, on="firm_ids")
    shares = shares.sort_values("volume_mean_share_pct", ascending=False)
    shares.to_csv(out / "firm_shares.csv", index=False, float_format="%.3f")
    print(shares.to_string(index=False))

    # Concentration market by market, and where each market falls against the EU guidelines
    by_market, summary, classes = [], [], []
    for merger in (MAIN_MERGER, CONTRAST_MERGER):
        for basis in ("volume", "value"):
            c = table.concentration(merger, basis)
            c["eu_hmg_indication"] = [eu_hmg_indication(r.hhi_post, r.delta_hhi, r.combined_share_pct,
                                                        r.max_party_share_pct) for r in c.itertuples()]
            c.insert(0, "basis", basis)
            c.insert(0, "merger", merger.name)
            by_market.append(c)
            avg = shares.set_index("firm_ids")[f"{'volume' if basis == 'volume' else 'value'}_mean_share_pct"]
            a, b = (float(avg[f]) for f in merger.firms)
            rest = [float(avg[f]) for f in avg.index if f not in merger.firms]
            pooled_pre, pooled_post = hhi(avg.to_numpy()), hhi([a + b] + rest)
            summary.append({
                "merger": merger.name, "basis": basis, "markets": len(c),
                "combined_share_pct_mean": c["combined_share_pct"].mean(),
                "combined_share_pct_min": c["combined_share_pct"].min(),
                "combined_share_pct_max": c["combined_share_pct"].max(),
                "markets_combined_share_25_plus": int((c["combined_share_pct"] >= 25).sum()),
                "hhi_pre_mean": c["hhi_pre"].mean(), "hhi_post_mean": c["hhi_post"].mean(),
                "delta_hhi_mean": c["delta_hhi"].mean(), "delta_hhi_min": c["delta_hhi"].min(),
                "delta_hhi_max": c["delta_hhi"].max(),
                "markets_in_eu_safe_harbour": int(c["eu_hmg_indication"].str.startswith("safe harbour").sum()),
                "hhi_pre_of_average_shares": pooled_pre, "hhi_post_of_average_shares": pooled_post,
                "delta_hhi_of_average_shares": pooled_post - pooled_pre,
            })
            counts = c["eu_hmg_indication"].value_counts().rename_axis("eu_hmg_indication").reset_index(name="markets")
            classes.append(counts.assign(merger=merger.name, basis=basis))
    pd.concat(by_market).to_csv(out / "concentration_by_market.csv", index=False, float_format="%.3f")
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "concentration_summary.csv", index=False, float_format="%.3f")
    print(summary.T.to_string())
    classes = pd.concat(classes)[["merger", "basis", "eu_hmg_indication", "markets"]]
    classes.to_csv(out / "eu_hmg_classification.csv", index=False)
    print(classes.to_string(index=False))

    # Critical loss for a 5% and a 10% SSNIP over a range of margins
    cl = pd.DataFrame([{"ssnip_pct": 100 * x, "margin_pct": 100 * m,
                        "critical_loss_pct": 100 * critical_loss(x, m),
                        "critical_elasticity": critical_elasticity(x, m)}
                       for x in (0.05, 0.10) for m in (0.2, 0.3, 0.4, 0.5, 0.6)])
    cl.to_csv(out / "critical_loss.csv", index=False, float_format="%.3f")
    print(cl.to_string(index=False))

    # GUPPIs from share-proportional diversion and stated margins (before any demand estimation)
    guppi = pd.concat([guppi_screen(products, MAIN_MERGER, m, o)
                       for o in STATED_OUTSIDE_DIVERSION for m in STATED_MARGINS], ignore_index=True)
    guppi = guppi[["firm_ids", "outside_diversion_pct", "stated_margin_pct", "diversion_to_partner_pct", "guppi_pct"]]
    guppi.to_csv(out / "guppi_screen.csv", index=False, float_format="%.3f")
    print(guppi.to_string(index=False))

    main_volume = pd.concat(by_market).query("merger == @MAIN_MERGER.name and basis == 'volume'")
    plot_concentration(main_volume, f"{MAIN_MERGER.name}: concentration in each of the 94 markets (volume shares)",
                       out / "concentration.png")


if __name__ == "__main__":
    main()
