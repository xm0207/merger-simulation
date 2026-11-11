"""Demand systems for the products of one market: logit, nested logit and random-coefficients logit.

Why a class hierarchy: the pricing equilibrium, the diversion ratios and the consumer-surplus
calculation never need to know which demand model they are using. They only ask three
questions, so `DemandModel` defines exactly those three abstract methods:

* `shares(prices)`: market shares s_j(p) of every product (the outside good is 1 - sum);
* `jacobian(prices)`: the matrix J[j, k] = ds_j / dp_k;
* `consumer_surplus(prices)`: expected consumer surplus per potential consumer, in the units
  of price (the log-sum formula divided by the marginal utility of income).

Elasticities and diversion ratios follow from the Jacobian, so the base class computes them
once for every model. A new demand model only has to implement the three methods.

Each model is built around the observed prices p0 and mean utilities delta0 recovered from
observed shares. A price change moves mean utility by alpha * (p - p0), where alpha < 0 is the
price coefficient. Holding the unobserved quality xi fixed in this way is what lets the model
predict shares at prices that were never observed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp

from .screening import diversion_matrix


class DemandModel(ABC):
    """Demand for the J products of one market as a function of their prices."""

    name = "demand"

    def __init__(self, prices0, delta0, alpha: float):
        self.prices0 = np.asarray(prices0, dtype=float)
        self.delta0 = np.asarray(delta0, dtype=float)
        if alpha >= 0:
            raise ValueError("the price coefficient alpha must be negative")
        self.alpha = float(alpha)

    def _delta(self, prices) -> np.ndarray:
        return self.delta0 + self.alpha * (np.asarray(prices, dtype=float) - self.prices0)

    @abstractmethod
    def shares(self, prices) -> np.ndarray:
        """Market shares of the J products at these prices."""

    @abstractmethod
    def jacobian(self, prices) -> np.ndarray:
        """J[j, k] = ds_j / dp_k at these prices."""

    @abstractmethod
    def consumer_surplus(self, prices) -> float:
        """Expected consumer surplus per potential consumer, in price units."""

    def elasticities(self, prices=None) -> np.ndarray:
        """E[j, k] = (ds_j / dp_k) * p_k / s_j; the diagonal holds own-price elasticities."""
        p = self.prices0 if prices is None else np.asarray(prices, dtype=float)
        return self.jacobian(p) * p[None, :] / self.shares(p)[:, None]

    def diversion(self, prices=None) -> np.ndarray:
        """Diversion ratios D[j, k] implied by the Jacobian (see `screening.diversion_matrix`)."""
        p = self.prices0 if prices is None else prices
        return diversion_matrix(self.jacobian(p))


class LogitDemand(DemandModel):
    """Plain logit: u_ij = delta_j + e_ij, with e_ij i.i.d. type I extreme value.

    s_j = exp(delta_j) / (1 + sum_k exp(delta_k)). Substitution is proportional to shares
    (independence of irrelevant alternatives), so the diversion from j to k is s_k / (1 - s_j)
    whatever the products' characteristics.
    """

    name = "logit"

    def shares(self, prices) -> np.ndarray:
        d = self._delta(prices)
        return np.exp(d - logsumexp(np.append(d, 0.0)))

    def jacobian(self, prices) -> np.ndarray:
        s = self.shares(prices)
        return self.alpha * (np.diag(s) - np.outer(s, s))

    def consumer_surplus(self, prices) -> float:
        return float(logsumexp(np.append(self._delta(prices), 0.0)) / -self.alpha)


class NestedLogitDemand(DemandModel):
    """One-level nested logit with nesting parameter rho in [0, 1).

    Products are grouped into nests g; the outside good is its own nest. With
    D_g = sum_{j in g} exp(delta_j / (1 - rho)):
        s_j = exp(delta_j / (1 - rho)) / D_g  *  D_g^(1 - rho) / (1 + sum_h D_h^(1 - rho)).
    rho = 0 is plain logit; as rho approaches 1, products in a nest become perfect substitutes
    for each other relative to anything outside the nest.
    """

    name = "nested logit"

    def __init__(self, prices0, delta0, alpha: float, rho: float, nests):
        super().__init__(prices0, delta0, alpha)
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        self.rho = float(rho)
        self.nests = np.asarray(nests)
        self._same_nest = (self.nests[:, None] == self.nests[None, :]).astype(float)
        self._labels = np.unique(self.nests)

    def _inclusive(self, prices):
        """log D_g for each nest, and each product's within-nest share s_{j|g}."""
        x = self._delta(prices) / (1.0 - self.rho)
        log_d = np.array([logsumexp(x[self.nests == g]) for g in self._labels])
        log_d_of_j = log_d[np.searchsorted(self._labels, self.nests)]
        return log_d, np.exp(x - log_d_of_j), log_d_of_j

    def shares(self, prices) -> np.ndarray:
        log_d, s_within, log_d_of_j = self._inclusive(prices)
        log_denominator = logsumexp(np.append((1.0 - self.rho) * log_d, 0.0))
        return s_within * np.exp((1.0 - self.rho) * log_d_of_j - log_denominator)

    def within_nest_shares(self, prices) -> np.ndarray:
        return self._inclusive(prices)[1]

    def jacobian(self, prices) -> np.ndarray:
        s = self.shares(prices)
        s_within = self.within_nest_shares(prices)
        r = self.rho
        inner = np.eye(len(s)) / (1 - r) - (r / (1 - r)) * self._same_nest * s_within[None, :] - s[None, :]
        return self.alpha * s[:, None] * inner

    def consumer_surplus(self, prices) -> float:
        log_d, _, _ = self._inclusive(prices)
        return float(logsumexp(np.append((1.0 - self.rho) * log_d, 0.0)) / -self.alpha)


class RandomCoefficientsDemand(DemandModel):
    """Random-coefficients (BLP) logit, integrated over a fixed set of simulated consumers.

    Consumer i's utility from product j is delta_j + mu_ij + e_ij, where
        mu_ij = x2_j' (Sigma nu_i + Pi d_i),
    nu_i are standard normal draws, d_i demographics and x2_j the characteristics with random
    coefficients (price among them). Shares are the weighted average of individual logit
    shares, so consumers who like similar characteristics substitute between similar products.
    The parameters come from pyblp; this class only evaluates the model, so that the same
    Bertrand solver handles all three demand systems.
    """

    name = "random coefficients"

    def __init__(self, prices0, delta0, alpha: float, x2, sigma, pi, nodes, demographics,
                 weights, price_column: int):
        super().__init__(prices0, delta0, alpha)
        self.x2 = np.asarray(x2, dtype=float)
        self.price_column = int(price_column)
        taste = np.asarray(sigma) @ np.asarray(nodes).T + np.asarray(pi) @ np.asarray(demographics).T  # K2 x I
        self.taste = taste
        self.weights = np.asarray(weights, dtype=float).ravel()
        # each consumer's marginal utility of price: alpha + their random price coefficient
        self.alpha_i = self.alpha + taste[self.price_column]

    def _mu(self, prices) -> np.ndarray:
        x2 = self.x2.copy()
        x2[:, self.price_column] = prices
        return x2 @ self.taste  # J x I

    def individual_shares(self, prices) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        u = self._delta(p)[:, None] + self._mu(p)
        log_denominator = logsumexp(np.vstack([u, np.zeros((1, u.shape[1]))]), axis=0)
        return np.exp(u - log_denominator[None, :])

    def shares(self, prices) -> np.ndarray:
        return self.individual_shares(prices) @ self.weights

    def jacobian(self, prices) -> np.ndarray:
        si = self.individual_shares(prices)
        wa = self.weights * self.alpha_i
        own = np.diag(si @ wa)
        cross = (si * wa[None, :]) @ si.T
        return own - cross

    def consumer_surplus(self, prices) -> float:
        """Weighted average of log-sum / -alpha_i. Consumers with alpha_i >= 0 are excluded."""
        p = np.asarray(prices, dtype=float)
        u = self._delta(p)[:, None] + self._mu(p)
        logsum = logsumexp(np.vstack([u, np.zeros((1, u.shape[1]))]), axis=0)
        ok = self.alpha_i < 0
        return float(np.sum(self.weights[ok] * logsum[ok] / -self.alpha_i[ok]))
