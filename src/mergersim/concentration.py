"""Market shares and concentration (HHI, delta HHI), computed in SQL with DuckDB.

The HHI is the sum of squared market shares, with shares in percent, so it runs from near 0
(atomistic market) to 10,000 (monopoly). A merger of firms with shares a and b raises it by
delta = (a + b)^2 - a^2 - b^2 = 2ab, if the merged firm keeps the parties' combined share.

Guideline benchmarks used for screening
---------------------------------------
* European Commission, Guidelines on the assessment of horizontal mergers, OJ C 31, 5.2.2004,
  section on market share and concentration levels. A combined share below 25% is an
  indication that the merger is compatible. Concerns are unlikely where the post-merger HHI is
  below 1,000, or is between 1,000 and 2,000 with a delta below 250, or is above 2,000 with a
  delta below 150 (para. 20), unless special circumstances apply, e.g. a party with a
  pre-merger share of 50% or more. These levels are an initial indicator only and create no
  presumption.
  A draft of new EU merger guidelines (April 2026) would replace this notice once adopted.
* CMA, Merger assessment guidelines (CMA129, March 2021, as updated on gov.uk): "The CMA does
  not apply any thresholds to market share, number of remaining competitors or on any other
  measure". Shares and the HHI are evidence on closeness of competition, not safe harbours.
  So the UK check here can only report the measures, not classify them.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import numpy as np
import pandas as pd

from .merger import Merger


def hhi(shares_pct: Sequence[float]) -> float:
    """Herfindahl-Hirschman index from firm shares in percent (they should sum to 100)."""
    s = np.asarray(shares_pct, dtype=float)
    return float(np.sum(s**2))


def delta_hhi(share_a_pct: float, share_b_pct: float) -> float:
    """Change in the HHI when two firms with these shares (in percent) merge: 2ab."""
    return 2.0 * share_a_pct * share_b_pct


def eu_hmg_indication(hhi_post: float, delta: float, combined_share_pct: float,
                      max_party_share_pct: float) -> str:
    """Where a merger falls against the EU Horizontal Merger Guidelines (2004) screening levels.

    Returns one short label. "outside safe harbours" means only that the soft safe harbours do
    not apply, so the merger needs a closer look; it is not a finding of harm.
    """
    if max_party_share_pct >= 50:
        return "outside safe harbours (party share >= 50%, a special circumstance)"
    if hhi_post < 1000:
        return "safe harbour: HHI < 1,000"
    if hhi_post <= 2000 and delta < 250:
        return "safe harbour: HHI 1,000-2,000 and delta < 250 (para. 20)"
    if hhi_post > 2000 and delta < 150:
        return "safe harbour: HHI > 2,000 and delta < 150 (para. 20)"
    if combined_share_pct < 25:
        return "outside HHI safe harbours, but combined share < 25%"
    return "outside safe harbours"


class ShareTable:
    """Product-level sales held in an in-memory DuckDB database, queried in SQL.

    Why a class: share and concentration tables are naturally GROUP BY queries, and several
    are needed (by firm, by market, pre and post merger, by volume and by value). The class
    registers the product data once and exposes three methods that each run one query.

    Volume shares use the `shares` column (quantity over market size; the market size cancels
    within a market). Value shares weight by price: p_j * s_j. All shares are of the products
    in the data, i.e. the outside good is not part of the market for screening purposes.
    """

    def __init__(self, products: pd.DataFrame):
        cols = ["market_ids", "firm_ids", "product_ids", "shares", "prices"]
        self.products = products[cols].copy()
        self.con = duckdb.connect()
        self.con.register("products", self.products)

    def _value_expr(self, basis: str) -> str:
        if basis == "volume":
            return "shares"
        if basis == "value":
            return "shares * prices"
        raise ValueError("basis must be 'volume' or 'value'")

    def firm_shares(self, basis: str = "volume") -> pd.DataFrame:
        """Each firm's share (%) of each market."""
        v = self._value_expr(basis)
        return self.con.execute(f"""
            SELECT market_ids, firm_ids,
                   100 * SUM({v}) / SUM(SUM({v})) OVER (PARTITION BY market_ids) AS share_pct
            FROM products
            GROUP BY market_ids, firm_ids
            ORDER BY market_ids, firm_ids
        """).df()

    def average_firm_shares(self, basis: str = "volume") -> pd.DataFrame:
        """Each firm's share (%) averaged over markets with equal weight, and its range.

        Equal weights because market sizes (population) are not in the data.
        """
        self.con.register("firm_shares", self.firm_shares(basis))
        return self.con.execute("""
            SELECT firm_ids, AVG(share_pct) AS mean_share_pct,
                   MIN(share_pct) AS min_share_pct, MAX(share_pct) AS max_share_pct,
                   COUNT(*) AS markets
            FROM firm_shares GROUP BY firm_ids ORDER BY mean_share_pct DESC
        """).df()

    def concentration(self, merger: Merger, basis: str = "volume") -> pd.DataFrame:
        """Pre- and post-merger HHI, delta and the parties' shares, one row per market.

        The post-merger HHI assumes the merged firm keeps the parties' combined share, the
        usual static convention; delta then equals 2ab exactly.
        """
        self.con.register("firm_shares", self.firm_shares(basis))
        firms = ", ".join(str(int(f)) for f in merger.firms)
        acquirer = int(merger.firms[0])
        return self.con.execute(f"""
            WITH post AS (
                SELECT market_ids,
                       CASE WHEN firm_ids IN ({firms}) THEN {acquirer} ELSE firm_ids END AS firm_ids,
                       SUM(share_pct) AS share_pct
                FROM firm_shares GROUP BY 1, 2
            ),
            pre_hhi AS (SELECT market_ids, SUM(share_pct * share_pct) AS hhi_pre FROM firm_shares GROUP BY 1),
            post_hhi AS (SELECT market_ids, SUM(share_pct * share_pct) AS hhi_post FROM post GROUP BY 1),
            parties AS (
                SELECT market_ids,
                       SUM(CASE WHEN firm_ids IN ({firms}) THEN share_pct ELSE 0 END) AS combined_share_pct,
                       MAX(CASE WHEN firm_ids IN ({firms}) THEN share_pct ELSE 0 END) AS max_party_share_pct,
                       MIN(CASE WHEN firm_ids IN ({firms}) THEN share_pct END) AS min_party_share_pct,
                       COUNT(*) AS firms_pre
                FROM firm_shares GROUP BY 1
            )
            SELECT pre_hhi.market_ids, firms_pre, combined_share_pct, max_party_share_pct,
                   min_party_share_pct, hhi_pre, hhi_post, hhi_post - hhi_pre AS delta_hhi
            FROM pre_hhi JOIN post_hhi USING (market_ids) JOIN parties USING (market_ids)
            ORDER BY market_ids
        """).df()
