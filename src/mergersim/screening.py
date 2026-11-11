"""First-pass screening tools that need only shares, margins and diversion ratios.

* Critical loss (Harris and Simons, 1989): a hypothetical monopolist that raises price by a
  fraction X, with a percentage margin M = (p - c)/p, breaks even if it loses a fraction
  CL = X / (X + M) of its sales. If the loss the demand system predicts ("actual loss") is
  smaller, the price rise is profitable and the candidate market passes the SSNIP test.
* Diversion ratio D_jk: of the sales product j loses when its price rises, the fraction that
  goes to product k. D_jk = -(ds_k/dp_j) / (ds_j/dp_j).
* Upward pricing pressure (Farrell and Shapiro, 2010): after a merger, a lost sale of product 1
  that diverts to the partner's product 2 is no longer fully lost. UPP_1 = D_12 (p_2 - c_2) - E c_1,
  where E is the claimed proportional cost saving on product 1.
* GUPPI (gross UPP index): UPP without efficiencies, as a share of price,
  GUPPI_1 = D_12 * m_2 * p_2 / p_1, with m_2 the partner's percentage margin.
  With several partner products: GUPPI_j = sum_k D_jk (p_k - c_k) / p_j.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def critical_loss(ssnip: float, margin: float) -> float:
    """Break-even critical loss X / (X + M) for a SSNIP X and percentage margin M (both fractions)."""
    if ssnip <= 0 or margin < 0:
        raise ValueError("ssnip must be positive and margin non-negative")
    return ssnip / (ssnip + margin)


def critical_elasticity(ssnip: float, margin: float) -> float:
    """Largest market elasticity at which the SSNIP still breaks even with linear demand: 1 / (X + M)."""
    return 1.0 / (ssnip + margin)


def diversion_matrix(jacobian: np.ndarray) -> np.ndarray:
    """D[j, k] = -J[k, j] / J[j, j], with J[j, k] = ds_j/dp_k. The diagonal is set to zero.

    Row j says where the sales lost by product j go; the row sums to one minus the diversion
    to the outside good.
    """
    J = np.asarray(jacobian, dtype=float)
    D = -J.T / np.diag(J)[:, None]
    np.fill_diagonal(D, 0.0)
    return D


def share_diversion(shares: Sequence[float], outside_diversion: float = 0.0) -> np.ndarray:
    """Diversion proportional to the other products' shares, with a stated fraction lost outside.

    D[j, k] = (1 - outside_diversion) * s_k / (sum of shares - s_j). This is the screening
    assumption used before any demand is estimated; it is what logit implies when
    `outside_diversion` equals the outside good's share of the remaining choices.
    """
    s = np.asarray(shares, dtype=float)
    D = (1.0 - outside_diversion) * s[None, :] / (s.sum() - s)[:, None]
    np.fill_diagonal(D, 0.0)
    return D


def upp(diversion_to_partner: float, partner_price: float, partner_cost: float,
        own_cost: float, efficiency: float = 0.0) -> float:
    """Farrell-Shapiro UPP for one product: D_12 (p_2 - c_2) - E c_1, in price units."""
    return diversion_to_partner * (partner_price - partner_cost) - efficiency * own_cost


def guppi(diversion_to_partner: float, partner_margin: float, partner_price: float,
          own_price: float) -> float:
    """GUPPI for one product: D_12 * m_2 * p_2 / p_1 (m_2 a fraction of p_2)."""
    return diversion_to_partner * partner_margin * partner_price / own_price


def guppi_vector(diversion: np.ndarray, prices: np.ndarray, costs: np.ndarray,
                 partner_mask: np.ndarray) -> np.ndarray:
    """GUPPI for every product j: sum over the partner's products k of D_jk (p_k - c_k) / p_j.

    `partner_mask[j, k]` is True when k belongs to the other merging party from j's point of
    view (and False for j's own firm and for non-merging firms).
    """
    D = np.asarray(diversion, dtype=float)
    margins = np.asarray(prices, dtype=float) - np.asarray(costs, dtype=float)
    return (D * partner_mask) @ margins / np.asarray(prices, dtype=float)


def partner_mask(firm_ids: Sequence, merging_firms: Sequence) -> np.ndarray:
    """mask[j, k] is True when j and k belong to different merging firms."""
    ids = np.asarray(firm_ids)
    merging = np.isin(ids, merging_firms)
    return merging[:, None] & merging[None, :] & (ids[:, None] != ids[None, :])
