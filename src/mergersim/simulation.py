"""Merger simulation across all markets for one estimated demand specification.

For each market:
1. take the demand model at observed prices p0;
2. recover marginal costs c from the pre-merger Bertrand-Nash first-order conditions;
3. change ownership (the merging firms' products now share one owner), optionally lower the
   merging firms' costs by a stated percentage, and solve for the new equilibrium prices p1;
4. compare shares, consumer surplus and margins before and after;
5. compute the compensating marginal-cost reductions (Werden, 1996) and GUPPIs at p0.

Market sizes (populations) are not in the data, so aggregates across markets weight every
market equally; within a market, averages over products weight by pre-merger quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .equilibrium import BertrandNash, compensating_cost_reductions
from .estimation import FittedDemand
from .merger import Merger, ownership_matrix
from .screening import guppi_vector, partner_mask


@dataclass
class SimulationResult:
    """Product-level and market-level outcomes of one merger under one demand specification.

    `products` has one row per product and market; `markets` one row per market with
    consumer surplus (per potential consumer, in price units) and spending on the products.
    """

    label: str
    merger: Merger
    products: pd.DataFrame
    markets: pd.DataFrame

    def _wavg(self, df: pd.DataFrame, col: str) -> float:
        return float(np.average(df[col], weights=df["share_pre"]))

    def by_firm(self) -> pd.DataFrame:
        """Quantity-weighted average price change (%) of each firm's products, over all markets."""
        rows = []
        for firm, g in self.products.groupby("firm_ids"):
            rows.append({"model": self.label, "firm_ids": firm, "merging": bool(g["merging"].iloc[0]),
                         "price_change_pct": self._wavg(g, "price_change_pct"),
                         "share_change_pct": 100 * (g["share_post"].sum() / g["share_pre"].sum() - 1)})
        return pd.DataFrame(rows)

    def by_product(self) -> pd.DataFrame:
        """Average over markets of each product's price change, margin, CMCR and GUPPI."""
        rows = []
        for (product, firm, merging), g in self.products.groupby(["product_ids", "firm_ids", "merging"]):
            positive = g[g["cost"] > 0]
            rows.append({
                "model": self.label, "product_ids": product, "firm_ids": firm, "merging": merging,
                "price_pre": g["price_pre"].mean(),
                "margin_pre_pct": self._wavg(g, "margin_pre_pct"),
                "price_change_pct": self._wavg(g, "price_change_pct"),
                "price_change_pct_min": g["price_change_pct"].min(),
                "price_change_pct_max": g["price_change_pct"].max(),
                "cmcr_pct": self._wavg(positive, "cmcr_pct") if merging and len(positive) else np.nan,
                "guppi_pct": self._wavg(g, "guppi_pct") if merging else np.nan,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Headline numbers: price effects, consumer surplus, profits, CMCR, GUPPI.

        Consumer surplus and profit changes are expressed as a percentage of pre-merger spending
        on all products (summed over markets): the level of logit consumer surplus depends on an
        arbitrary normalisation of the outside good's utility, but its change does not.
        CMCRs are averaged only over product-markets with positive recovered costs, because a
        proportional cost reduction is meaningless for a zero or negative cost.
        """
        p = self.products
        m = p[p["merging"]]
        m_pos = m[m["cost"] > 0]
        markets = self.markets
        spending = markets["spending_pre"].sum()
        return {
            "model": self.label,
            "merger": self.merger.name,
            "markets": len(markets),
            "merging_price_change_pct": self._wavg(m, "price_change_pct"),
            "rivals_price_change_pct": self._wavg(p[~p["merging"]], "price_change_pct"),
            "all_price_change_pct": self._wavg(p, "price_change_pct"),
            "max_product_market_price_change_pct": float(m["price_change_pct"].max()),
            "consumer_surplus_change_pct_of_spending": float(100 * markets["cs_change"].sum() / spending),
            "merging_profit_change_pct_of_spending": float(100 * markets["merging_profit_change"].sum() / spending),
            "rivals_profit_change_pct_of_spending": float(100 * markets["rivals_profit_change"].sum() / spending),
            "merging_margin_pre_pct": self._wavg(m, "margin_pre_pct"),
            "merging_guppi_pct": self._wavg(m, "guppi_pct"),
            "merging_cmcr_pct": self._wavg(m_pos, "cmcr_pct"),
            "merging_cmcr_pct_median": float(m_pos["cmcr_pct"].median()),
            "merging_nonpositive_costs": int(len(m) - len(m_pos)),
            "all_nonpositive_costs": int((p["cost"] <= 0).sum()),
            "max_foc_residual_pre": float(markets["foc_residual_pre"].max()),
        }


class MergerSimulator:
    """Simulates one merger, market by market, under one estimated demand specification.

    Why a class: the fitted demand and the merger are fixed for a whole run, while the
    per-market work and the optional efficiency (cost reduction) vary. Public interface:
    `run_market(market_id)` and `run()`.
    """

    def __init__(self, fit: FittedDemand, merger: Merger):
        self.fit = fit
        self.merger = merger

    def run_market(self, market_id, cost_reduction: float = 0.0) -> tuple[pd.DataFrame, dict]:
        rows = self.fit.rows(market_id)
        info = self.fit.products.iloc[rows]
        firm_ids = info["firm_ids"].to_numpy()
        demand = self.fit.market(market_id)
        p0 = demand.prices0
        merging = self.merger.involves(firm_ids)

        pre = BertrandNash(demand, self.merger.ownership(firm_ids, post=False))
        costs = pre.recover_costs(p0)
        post = BertrandNash(demand, self.merger.ownership(firm_ids, post=True))
        costs_post = np.where(merging, costs * (1.0 - cost_reduction), costs)
        p1 = post.solve(costs_post)

        s0, s1 = demand.shares(p0), demand.shares(p1)
        c_star = compensating_cost_reductions(demand, post.ownership, costs, merging)
        guppi = guppi_vector(demand.diversion(p0), p0, costs, partner_mask(firm_ids, self.merger.firms))

        products = pd.DataFrame({
            "market_ids": market_id,
            "product_ids": info["product_ids"].to_numpy(),
            "firm_ids": firm_ids,
            "merging": merging,
            "price_pre": p0,
            "price_post": p1,
            "price_change_pct": 100 * (p1 / p0 - 1),
            "share_pre": s0,
            "share_post": s1,
            "cost": costs,
            "margin_pre_pct": 100 * (p0 - costs) / p0,
            "cmcr_pct": np.where(merging, 100 * (costs - c_star) / costs, np.nan),
            "guppi_pct": np.where(merging, 100 * guppi, np.nan),
        })
        cs0, cs1 = demand.consumer_surplus(p0), demand.consumer_surplus(p1)
        market = {
            "market_ids": market_id,
            "cs_pre": cs0,
            "cs_post": cs1,
            "cs_change": cs1 - cs0,
            "spending_pre": float(p0 @ s0),
            "merging_profit_change": float(((p1 - costs_post) * s1)[merging].sum() - ((p0 - costs) * s0)[merging].sum()),
            "rivals_profit_change": float(((p1 - costs) * s1)[~merging].sum() - ((p0 - costs) * s0)[~merging].sum()),
            "foc_residual_pre": float(np.max(np.abs(pre.foc_residual(p0, costs)))),
            "negative_costs": int((costs <= 0).sum()),
        }
        return products, market

    def run(self, cost_reduction: float = 0.0) -> SimulationResult:
        """Simulate every market; `cost_reduction` lowers the merging firms' marginal costs (0.05 = 5%)."""
        product_frames, market_rows = [], []
        for market_id in self.fit.market_ids:
            products, market = self.run_market(market_id, cost_reduction)
            product_frames.append(products)
            market_rows.append(market)
        return SimulationResult(self.fit.label, self.merger, pd.concat(product_frames, ignore_index=True),
                                pd.DataFrame(market_rows))


def hypothetical_monopolist_test(fit: FittedDemand, ssnip: float = 0.05) -> pd.DataFrame:
    """SSNIP test of the candidate market "all products in the data, in one market".

    A hypothetical monopolist of every product raises all prices by `ssnip`. Costs are those
    recovered under the observed (pre-merger) ownership. For each market, reports the actual
    loss of quantity, the critical loss X / (X + M) at the quantity-weighted margin M, and
    whether the price rise is profitable (profit is computed directly, not from the formula).
    """
    rows = []
    for market_id in fit.market_ids:
        rows_idx = fit.rows(market_id)
        firm_ids = fit.products["firm_ids"].to_numpy()[rows_idx]
        demand = fit.market(market_id)
        p0 = demand.prices0
        costs = BertrandNash(demand, ownership_matrix(firm_ids)).recover_costs(p0)
        s0, s1 = demand.shares(p0), demand.shares(p0 * (1 + ssnip))
        margin = float(np.sum((p0 - costs) * s0) / np.sum(p0 * s0))
        actual_loss = 1 - s1.sum() / s0.sum()
        profit0, profit1 = np.sum((p0 - costs) * s0), np.sum((p0 * (1 + ssnip) - costs) * s1)
        rows.append({"market_ids": market_id, "ssnip": ssnip, "margin": margin,
                     "critical_loss": ssnip / (ssnip + margin), "actual_loss": actual_loss,
                     "profit_change_pct": 100 * (profit1 / profit0 - 1), "profitable": bool(profit1 > profit0)})
    return pd.DataFrame(rows).assign(model=fit.label)


def demand_diagnostics(fit: FittedDemand) -> pd.DataFrame:
    """Own- and cross-price elasticities and Bertrand-Nash margins for every product and market.

    Margins use costs recovered under the observed ownership, so they are what each demand
    model implies about pricing before any merger. `cross_elasticity_median` is the median of
    product j's elasticity with respect to the other products' prices in its market.
    """
    frames = []
    for market_id in fit.market_ids:
        rows = fit.rows(market_id)
        firm_ids = fit.products["firm_ids"].to_numpy()[rows]
        demand = fit.market(market_id)
        p0 = demand.prices0
        E = demand.elasticities(p0)
        off = ~np.eye(len(p0), dtype=bool)
        costs = BertrandNash(demand, ownership_matrix(firm_ids)).recover_costs(p0)
        frames.append(pd.DataFrame({
            "market_ids": market_id,
            "product_ids": fit.products["product_ids"].to_numpy()[rows],
            "firm_ids": firm_ids,
            "price": p0,
            "share": demand.shares(p0),
            "own_elasticity": np.diag(E),
            "cross_elasticity_median": np.nanmedian(np.where(off, E, np.nan), axis=1),
            "diversion_to_outside": 1 - demand.diversion(p0).sum(axis=1),
            "cost": costs,
            "margin_pct": 100 * (p0 - costs) / p0,
        }))
    return pd.concat(frames, ignore_index=True).assign(model=fit.label)
