import numpy as np
import pandas as pd
import pytest

from conftest import ALPHA, DELTA, NESTS, PRICES, finite_difference_jacobian
from mergersim.demand import LogitDemand, NestedLogitDemand, RandomCoefficientsDemand
from mergersim.estimation import berry_inversion, within_nest_shares


def test_logit_shares_sum_below_one(logit):
    s = logit.shares(PRICES)
    assert np.all(s > 0) and s.sum() < 1
    assert np.allclose(s, np.exp(DELTA) / (1 + np.exp(DELTA).sum()))


def test_berry_inversion_recovers_mean_utilities(logit):
    s = logit.shares(PRICES)
    frame = pd.DataFrame({"shares": s, "outside_share": 1 - s.sum()})
    assert np.allclose(berry_inversion(frame), DELTA)


def test_nested_inversion_recovers_mean_utilities(nested):
    s = nested.shares(PRICES)
    frame = pd.DataFrame({"market_ids": 0, "nest": NESTS, "shares": s, "outside_share": 1 - s.sum()})
    delta = berry_inversion(frame) - nested.rho * np.log(within_nest_shares(frame, "nest"))
    assert np.allclose(delta, DELTA)


def test_logit_elasticities_match_closed_form(logit):
    s = logit.shares(PRICES)
    E = logit.elasticities()
    assert np.allclose(np.diag(E), ALPHA * PRICES * (1 - s))
    for j in range(5):
        for k in range(5):
            if j != k:
                assert E[j, k] == pytest.approx(-ALPHA * PRICES[k] * s[k])


@pytest.mark.parametrize("model", ["logit", "nested", "rc"])
def test_jacobian_matches_finite_differences(model, request):
    demand = request.getfixturevalue(model)
    p = PRICES * 1.03  # away from the observed prices too
    assert np.allclose(demand.jacobian(p), finite_difference_jacobian(demand, p), atol=1e-7)


@pytest.mark.parametrize("model", ["logit", "nested", "rc"])
def test_roys_identity(model, request):
    """d CS / d p_j = -s_j: a price rise costs consumers their quantity times the rise."""
    demand = request.getfixturevalue(model)
    h = 1e-6
    for j in range(5):
        up, down = PRICES.copy(), PRICES.copy()
        up[j] += h
        down[j] -= h
        grad = (demand.consumer_surplus(up) - demand.consumer_surplus(down)) / (2 * h)
        assert grad == pytest.approx(-demand.shares(PRICES)[j], rel=1e-5)


def test_nested_logit_with_rho_zero_is_logit(logit):
    nested0 = NestedLogitDemand(PRICES, DELTA, ALPHA, 0.0, NESTS)
    assert np.allclose(nested0.shares(PRICES), logit.shares(PRICES))
    assert np.allclose(nested0.jacobian(PRICES), logit.jacobian(PRICES))


def test_rc_without_heterogeneity_is_logit(logit):
    x2 = np.column_stack([np.ones(5), PRICES])
    rc0 = RandomCoefficientsDemand(PRICES, DELTA, ALPHA, x2, np.zeros((2, 2)), np.zeros((2, 1)),
                                   np.random.default_rng(1).standard_normal((50, 2)), np.ones((50, 1)),
                                   np.full(50, 0.02), price_column=1)
    assert np.allclose(rc0.shares(PRICES), logit.shares(PRICES))
    assert rc0.consumer_surplus(PRICES) == pytest.approx(logit.consumer_surplus(PRICES))


def test_nesting_raises_within_nest_substitution(logit, nested):
    """With rho > 0, a product loses relatively more sales to its own nest than logit predicts."""
    D_logit, D_nested = logit.diversion(), nested.diversion()
    same = (NESTS[:, None] == NESTS[None, :]) & ~np.eye(5, dtype=bool)
    assert (D_nested[same] / D_logit[same]).mean() > (D_nested[~same & ~np.eye(5, dtype=bool)] / D_logit[~same & ~np.eye(5, dtype=bool)]).mean()


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        LogitDemand(PRICES, DELTA, alpha=0.5)
    with pytest.raises(ValueError):
        NestedLogitDemand(PRICES, DELTA, ALPHA, 1.0, NESTS)
