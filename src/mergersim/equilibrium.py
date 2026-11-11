"""Multiproduct Bertrand-Nash pricing: recover costs before a merger, solve prices after it.

Each firm f sets the prices of its products to maximise sum_{j in f} (p_j - c_j) s_j(p).
The first-order condition for product j is
    s_j + sum_k Omega[j, k] * (ds_k / dp_j) * (p_k - c_k) = 0,
or, stacking all products, s(p) + (Omega o J(p)^T)(p - c) = 0, where Omega is the ownership
matrix and o is the element-wise product. Rearranged, the markup is
    p - c = Delta(p)^{-1} s(p),  with Delta(p) = -(Omega o J(p)^T).

Before the merger, observed prices are assumed to be an equilibrium, so this equation gives
marginal costs. After the merger, Omega has more ones and the same costs give higher prices.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .demand import DemandModel


class BertrandNash:
    """The pricing game of one market under one ownership structure.

    Why a class: cost recovery, the equilibrium solver and the first-order-condition check all
    need the same pair (demand model, ownership matrix); holding the pair makes each call short.
    Public interface: `markups`, `recover_costs`, `foc_residual`, `solve`.
    """

    def __init__(self, demand: DemandModel, ownership: np.ndarray):
        self.demand = demand
        self.ownership = np.asarray(ownership, dtype=float)

    def markups(self, prices) -> np.ndarray:
        """p - c implied by the first-order conditions at these prices: Delta(p)^{-1} s(p)."""
        delta = -(self.ownership * self.demand.jacobian(prices).T)
        return np.linalg.solve(delta, self.demand.shares(prices))

    def recover_costs(self, prices) -> np.ndarray:
        """Marginal costs that make these prices a Bertrand-Nash equilibrium."""
        return np.asarray(prices, dtype=float) - self.markups(prices)

    def foc_residual(self, prices, costs) -> np.ndarray:
        """s(p) + (Omega o J^T)(p - c); zero at an equilibrium."""
        p = np.asarray(prices, dtype=float)
        jt = self.demand.jacobian(p).T
        return self.demand.shares(p) + (self.ownership * jt) @ (p - np.asarray(costs, dtype=float))

    def markup_residual(self, prices, costs) -> np.ndarray:
        """p - c - Delta(p)^{-1} s(p), in price units; zero at an equilibrium.

        This form is used to check solutions because it stays well scaled when a share is
        tiny: the plain first-order condition is close to zero for any product priced so
        high that it sells nothing, which would let a solver stop at a meaningless point.
        """
        p = np.asarray(prices, dtype=float)
        return p - np.asarray(costs, dtype=float) - self.markups(p)

    def _iterate(self, costs, start, damping: float, tol: float, max_iter: int):
        p = start.copy()
        try:
            for _ in range(max_iter):
                step = costs + self.markups(p) - p
                if not np.all(np.isfinite(step)):
                    return None
                p = p + damping * step
                if np.max(np.abs(step)) < tol:
                    return p
        except np.linalg.LinAlgError:
            return None
        return None

    def solve(self, costs, start=None, tol: float = 1e-12, max_iter: int = 2000,
              rel_tol: float = 1e-9) -> np.ndarray:
        """Equilibrium prices for these marginal costs.

        Three methods are tried in turn, starting from `start` (default: the observed prices):
        1. the markup iteration p <- c + Delta(p)^{-1} s(p), which is fast but can cycle when
           one firm has a very large share;
        2. a Newton-type root finder (MINPACK hybrid) on the markup residual;
        3. the markup iteration with damping 0.5, then 0.2.
        A candidate is accepted only if the markup residual is below `rel_tol` times the
        largest price, i.e. every first-order condition holds.
        """
        c = np.asarray(costs, dtype=float)
        p0 = self.demand.prices0.copy() if start is None else np.asarray(start, dtype=float).copy()

        def accept(p):
            if p is None or not np.all(np.isfinite(p)):
                return False
            try:
                return np.max(np.abs(self.markup_residual(p, c))) < rel_tol * np.max(np.abs(p))
            except np.linalg.LinAlgError:
                return False

        p = self._iterate(c, p0, 1.0, tol, max_iter)
        if accept(p):
            return p
        try:
            sol = optimize.root(self.markup_residual, p0, args=(c,), method="hybr", options={"xtol": tol})
            if accept(sol.x):
                return sol.x
        except np.linalg.LinAlgError:
            pass
        for damping in (0.5, 0.2):
            p = self._iterate(c, p0, damping, tol, 10 * max_iter)
            if accept(p):
                return p
        raise RuntimeError("Bertrand-Nash solver did not converge")


def compensating_cost_reductions(demand: DemandModel, ownership_post: np.ndarray,
                                 costs, merging: np.ndarray) -> np.ndarray:
    """Werden (1996): merging products' marginal costs that keep post-merger prices at p0.

    At the observed prices p0, the non-merging firms' first-order conditions still hold (their
    ownership has not changed), so only the merging firms' conditions need new costs:
        c*_M = p0_M - [Delta_post(p0)]_MM^{-1} s_M(p0).
    Returns c* for every product (unchanged for non-merging products). The proportional
    compensating marginal-cost reduction (CMCR) of product j is (c_j - c*_j) / c_j.
    """
    p0 = demand.prices0
    m = np.asarray(merging, dtype=bool)
    delta_post = -(np.asarray(ownership_post) * demand.jacobian(p0).T)
    c_star = np.asarray(costs, dtype=float).copy()
    c_star[m] = p0[m] - np.linalg.solve(delta_post[np.ix_(m, m)], demand.shares(p0)[m])
    return c_star
