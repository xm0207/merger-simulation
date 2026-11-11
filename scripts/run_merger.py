"""Simulate the hypothetical merger of Firm 3 and Firm 4 under logit, nested logit and BLP demand.

For each of the 94 markets: recover marginal costs from pre-merger Bertrand-Nash pricing,
change ownership, solve the post-merger equilibrium, and compute consumer-surplus changes,
compensating marginal-cost reductions (Werden, 1996) and GUPPIs. Also runs the hypothetical
monopolist (SSNIP) test, sensitivity checks, and a cross-check of the BLP simulation against
pyblp's own post-estimation routines.

Writes to results/: merger_summary.csv, merger_by_product.csv, merger_by_firm.csv, hmt.csv,
sensitivity.csv, pyblp_check.csv, price_effects.png and efficiencies.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mergersim import LogitFit, MergerSimulator, hypothetical_monopolist_test
from mergersim.data import load_cereal, load_cereal_agents
from mergersim.report import plot_efficiencies, plot_price_effects
from mergersim.specs import (CENTRAL_RHO, CONTRAST_MERGER, MAIN_MERGER, RHO_GRID, central_models, cereal_logit,
                             cereal_nested_logit)

COST_REDUCTIONS = (0.0, 0.025, 0.05, 0.075, 0.10)
MARKET_SIZE_FACTORS = (0.75, 1.5)
SUMMARY_COLUMNS = ["merging_price_change_pct", "rivals_price_change_pct", "all_price_change_pct",
                   "consumer_surplus_change_pct_of_spending", "merging_cmcr_pct", "merging_guppi_pct"]


def rescale_market_size(products: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Shares if the potential market were `factor` times as large (same quantities)."""
    out = products.copy()
    out["shares"] = out["shares"] / factor
    out["outside_share"] = 1 - out.groupby("market_ids")["shares"].transform("sum")
    if (out["outside_share"] <= 0).any():
        raise ValueError("market size too small: inside shares sum to one or more")
    return out


def sensitivity_row(check: str, setting: str, fit, result) -> dict:
    s = result.summary()
    row = {"check": check, "setting": setting, "model": fit.label, "alpha": fit.alpha,
           "rho": getattr(fit, "rho", np.nan)}
    row.update({k: s[k] for k in SUMMARY_COLUMNS})
    return row


def pyblp_check(rc_fit, main_result) -> pd.DataFrame:
    """Compare this package's BLP simulation with pyblp's compute_costs/prices/surpluses/elasticities."""
    r = rc_fit.results
    costs = r.compute_costs()
    merger_ids = MAIN_MERGER.post_firm_ids(r.problem.products.firm_ids.ravel())
    prices_post = r.compute_prices(firm_ids=merger_ids[:, None], costs=costs)
    cs_pre = pd.Series(r.compute_consumer_surpluses().ravel(), index=r.problem.unique_market_ids.ravel())
    cs_post = pd.Series(r.compute_consumer_surpluses(prices=prices_post).ravel(), index=r.problem.unique_market_ids.ravel())
    own = r.extract_diagonals(r.compute_elasticities()).ravel()
    mine = main_result.products
    markets = main_result.markets.set_index("market_ids")
    own_mine = np.concatenate([np.diag(rc_fit.market(m).elasticities()) for m in rc_fit.market_ids])
    return pd.DataFrame([
        {"quantity": "marginal costs", "max_abs_difference": np.max(np.abs(costs.ravel() - mine["cost"].to_numpy()))},
        {"quantity": "post-merger prices", "max_abs_difference": np.max(np.abs(prices_post.ravel() - mine["price_post"].to_numpy()))},
        {"quantity": "own-price elasticities", "max_abs_difference": np.max(np.abs(own - own_mine))},
        {"quantity": "consumer surplus, pre-merger", "max_abs_difference": np.max(np.abs(cs_pre - markets["cs_pre"].reindex(cs_pre.index)))},
        {"quantity": "consumer surplus, post-merger", "max_abs_difference": np.max(np.abs(cs_post - markets["cs_post"].reindex(cs_post.index)))},
    ]).assign(typical_magnitude=[float(np.median(np.abs(costs))), float(np.median(prices_post)), float(np.median(np.abs(own))),
                                 float(cs_pre.median()), float(cs_post.median())])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    products, agents = load_cereal(), load_cereal_agents()
    models = central_models(products, agents)

    # 1. Main merger (and the contrast merger) under the three demand models
    summaries, by_product, by_firm, main_results = [], [], [], {}
    for merger in (MAIN_MERGER, CONTRAST_MERGER):
        for fit in models:
            result = MergerSimulator(fit, merger).run()
            summaries.append(result.summary())
            if merger is MAIN_MERGER:
                main_results[fit.label] = result
                by_product.append(result.by_product())
                by_firm.append(result.by_firm())
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "merger_summary.csv", index=False, float_format="%.4f")
    print(summary.T.to_string())
    by_product = pd.concat(by_product, ignore_index=True)
    by_product.to_csv(out / "merger_by_product.csv", index=False, float_format="%.4f")
    print(by_product[by_product["merging"]].to_string(index=False))
    by_firm = pd.concat(by_firm, ignore_index=True)
    by_firm.to_csv(out / "merger_by_firm.csv", index=False, float_format="%.4f")
    print(by_firm.to_string(index=False))

    # 2. Hypothetical monopolist test of "all 24 brands in a city-quarter"
    hmt = []
    for fit in models:
        for ssnip in (0.05, 0.10):
            h = hypothetical_monopolist_test(fit, ssnip)
            hmt.append({"model": fit.label, "ssnip_pct": 100 * ssnip, "markets": len(h),
                        "markets_profitable": int(h["profitable"].sum()),
                        "margin_mean_pct": 100 * h["margin"].mean(),
                        "critical_loss_mean_pct": 100 * h["critical_loss"].mean(),
                        "actual_loss_mean_pct": 100 * h["actual_loss"].mean(),
                        "actual_loss_max_pct": 100 * h["actual_loss"].max(),
                        "profit_change_mean_pct": h["profit_change_pct"].mean()})
    hmt = pd.DataFrame(hmt)
    hmt.to_csv(out / "hmt.csv", index=False, float_format="%.3f")
    print(hmt.to_string(index=False))

    # 3. Sensitivity checks on the main merger
    rows = []
    for rho in RHO_GRID:
        fit = cereal_nested_logit(products, rho)
        rows.append(sensitivity_row("nesting parameter", f"rho = {rho:g}", fit, MergerSimulator(fit, MAIN_MERGER).run()))
    logit = models[0]
    se = logit.result.se["prices"]
    for sign, name in ((+1, "less elastic"), (-1, "more elastic")):
        alpha = logit.alpha + sign * 1.96 * se
        fit = LogitFit(f"Logit (alpha = {alpha:.1f})", products, alpha, logit.result)
        rows.append(sensitivity_row("logit price coefficient", f"95% CI bound, {name}", fit,
                                    MergerSimulator(fit, MAIN_MERGER).run()))
    for factor in MARKET_SIZE_FACTORS:
        scaled = rescale_market_size(products, factor)
        for fit in (cereal_logit(scaled), cereal_nested_logit(scaled, CENTRAL_RHO)):
            rows.append(sensitivity_row("market size", f"x{factor:g}", fit, MergerSimulator(fit, MAIN_MERGER).run()))
    efficiency_rows = []
    for fit in models:
        for cr in COST_REDUCTIONS:
            result = main_results[fit.label] if cr == 0 else MergerSimulator(fit, MAIN_MERGER).run(cost_reduction=cr)
            row = sensitivity_row("merger efficiencies", f"merging firms' costs -{100 * cr:g}%", fit, result)
            if cr > 0:  # CMCR and GUPPI are measured at pre-merger costs, so they do not change with efficiencies
                row.update(merging_cmcr_pct=np.nan, merging_guppi_pct=np.nan)
            rows.append(row)
            efficiency_rows.append({"model": fit.label, "cost_reduction_pct": 100 * cr,
                                    "merging_price_change_pct": result.summary()["merging_price_change_pct"]})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "sensitivity.csv", index=False, float_format="%.4f")
    print(sens.to_string(index=False))

    # 4. Cross-check the BLP simulation against pyblp
    check = pyblp_check(models[-1], main_results[models[-1].label])
    check.to_csv(out / "pyblp_check.csv", index=False, float_format="%.3e")
    print(check.to_string(index=False))

    # 5. Charts
    merging = by_product[by_product["merging"]]
    rivals = pd.DataFrame([{"model": label, "price_change_pct": res.summary()["rivals_price_change_pct"]}
                           for label, res in main_results.items()])
    plot_price_effects(merging, rivals, f"{MAIN_MERGER.name}: simulated price changes by brand (no efficiencies)",
                       out / "price_effects.png")
    cmcr = {label: res.summary()["merging_cmcr_pct"] for label, res in main_results.items()}
    plot_efficiencies(pd.DataFrame(efficiency_rows), cmcr,
                      f"{MAIN_MERGER.name}: price effect against cost savings (tick = average CMCR)",
                      out / "efficiencies.png")


if __name__ == "__main__":
    main()
