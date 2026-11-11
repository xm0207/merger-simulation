"""Small synthetic fixtures, so that every test runs offline and without pyblp's data files."""

import numpy as np
import pandas as pd
import pytest

from mergersim.demand import LogitDemand, NestedLogitDemand, RandomCoefficientsDemand

# One toy market: five products, three firms (A owns 0-1, B owns 2-3, C owns 4).
PRICES = np.array([1.0, 1.2, 0.9, 1.1, 1.3])
DELTA = np.array([0.5, 0.8, 0.2, 0.6, 0.9])
FIRMS = np.array([1, 1, 2, 2, 3])
NESTS = np.array([0, 1, 0, 1, 1])
ALPHA = -2.0


@pytest.fixture
def logit():
    return LogitDemand(PRICES, DELTA, ALPHA)


@pytest.fixture
def nested():
    return NestedLogitDemand(PRICES, DELTA, ALPHA, 0.6, NESTS)


@pytest.fixture
def rc():
    """Random coefficients on a constant and on price, one demographic, 200 simulated consumers."""
    rng = np.random.default_rng(0)
    n = 200
    x2 = np.column_stack([np.ones(5), PRICES])
    sigma = np.diag([0.5, 0.3])
    pi = np.array([[0.2], [0.4]])
    nodes = rng.standard_normal((n, 2))
    demographics = rng.standard_normal((n, 1))
    weights = np.full(n, 1.0 / n)
    return RandomCoefficientsDemand(PRICES, DELTA, ALPHA, x2, sigma, pi, nodes, demographics, weights, price_column=1)


def make_products(n_markets: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic product data in the layout of the pyblp files: logit shares from known utilities."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_markets):
        prices = PRICES * (1 + 0.1 * rng.standard_normal(5))
        delta = DELTA + 0.2 * rng.standard_normal(5) + ALPHA * (prices - PRICES)
        shares = np.exp(delta) / (1 + np.exp(delta).sum())
        for j in range(5):
            rows.append({"market_ids": f"M{t}", "firm_ids": int(FIRMS[j]), "product_ids": f"P{j}",
                         "prices": prices[j], "shares": shares[j], "nest": int(NESTS[j])})
    products = pd.DataFrame(rows)
    products["outside_share"] = 1 - products.groupby("market_ids")["shares"].transform("sum")
    return products


@pytest.fixture
def products():
    return make_products()


def finite_difference_jacobian(demand, prices, h=1e-6):
    """J[j, k] = ds_j / dp_k by central differences."""
    J = np.zeros((len(prices), len(prices)))
    for k in range(len(prices)):
        up, down = prices.copy(), prices.copy()
        up[k] += h
        down[k] -= h
        J[:, k] = (demand.shares(up) - demand.shares(down)) / (2 * h)
    return J
