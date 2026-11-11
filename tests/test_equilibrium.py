import numpy as np
import pytest

from conftest import ALPHA, FIRMS, PRICES
from mergersim import BertrandNash, Merger, compensating_cost_reductions, ownership_matrix
from mergersim.demand import LogitDemand

MERGER = Merger("A+B", (1, 2))


def test_ownership_matrices():
    assert np.array_equal(ownership_matrix([1, 1, 2]), [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    post = MERGER.ownership(FIRMS, post=True)
    assert post[0, 3] == 1 and post[0, 4] == 0
    assert np.array_equal(MERGER.post_firm_ids(FIRMS), [1, 1, 1, 1, 3])
    assert np.array_equal(MERGER.involves(FIRMS), [True, True, True, True, False])


def test_single_product_logit_markup_closed_form(logit):
    """With one product per firm, the logit markup is 1 / (-alpha (1 - s_j))."""
    game = BertrandNash(logit, np.eye(5))
    s = logit.shares(PRICES)
    assert np.allclose(game.markups(PRICES), 1 / (-ALPHA * (1 - s)))


@pytest.mark.parametrize("model", ["logit", "nested", "rc"])
def test_solver_reproduces_observed_prices_from_recovered_costs(model, request):
    demand = request.getfixturevalue(model)
    game = BertrandNash(demand, ownership_matrix(FIRMS))
    costs = game.recover_costs(PRICES)
    assert np.max(np.abs(game.foc_residual(PRICES, costs))) < 1e-12
    # start the solver somewhere else; it must find the observed prices again
    solved = game.solve(costs, start=PRICES * 1.3)
    assert np.allclose(solved, PRICES, atol=1e-9)


@pytest.mark.parametrize("model", ["logit", "nested", "rc"])
def test_merger_raises_prices(model, request):
    demand = request.getfixturevalue(model)
    costs = BertrandNash(demand, ownership_matrix(FIRMS)).recover_costs(PRICES)
    post = BertrandNash(demand, MERGER.ownership(FIRMS)).solve(costs)
    assert np.all(post[:4] > PRICES[:4])      # merging products
    assert np.all(post[4:] >= PRICES[4:])     # the rival responds with a (weakly) higher price


@pytest.mark.parametrize("model", ["logit", "nested", "rc"])
def test_compensating_cost_reductions_restore_prices(model, request):
    demand = request.getfixturevalue(model)
    costs = BertrandNash(demand, ownership_matrix(FIRMS)).recover_costs(PRICES)
    merging = MERGER.involves(FIRMS)
    post = BertrandNash(demand, MERGER.ownership(FIRMS))
    c_star = compensating_cost_reductions(demand, post.ownership, costs, merging)
    assert np.all(c_star[merging] < costs[merging])
    assert np.array_equal(c_star[~merging], costs[~merging])
    assert np.allclose(post.solve(c_star), PRICES, atol=1e-9)


def test_werden_closed_form_for_two_single_product_logit_firms():
    """Werden (1996) for two symmetric single-product firms: CMCR = m D / ((1 - m)(1 - D)),
    with m the pre-merger percentage margin and D the diversion ratio between them."""
    prices = np.array([1.0, 1.0, 1.0])
    demand = LogitDemand(prices, np.array([0.3, 0.3, 0.1]), -3.0)
    costs = BertrandNash(demand, np.eye(3)).recover_costs(prices)
    merger = Merger("1+2", (1, 2))
    firms = np.array([1, 2, 3])
    c_star = compensating_cost_reductions(demand, merger.ownership(firms), costs, merger.involves(firms))
    m = (prices[0] - costs[0]) / prices[0]
    D = demand.diversion()[0, 1]
    assert (costs[0] - c_star[0]) / costs[0] == pytest.approx(m * D / ((1 - m) * (1 - D)))


def test_solver_handles_dominant_firm():
    """A firm with most of the market makes the plain markup iteration cycle; the solver must still converge."""
    prices = np.array([0.05, 0.06, 0.12, 0.13, 0.14])
    demand = LogitDemand(prices, np.array([2.0, 1.8, 0.0, -0.5, -0.4]), -30.0)  # firm 1 sells 80%
    firms = np.array([1, 1, 2, 3, 4])
    costs = BertrandNash(demand, ownership_matrix(firms)).recover_costs(prices)
    post = BertrandNash(demand, Merger("3+4", (3, 4)).ownership(firms))
    assert post._iterate(costs, prices, 1.0, 1e-12, 500) is None  # the undamped iteration cycles here
    p1 = post.solve(costs)
    assert np.max(np.abs(post.markup_residual(p1, costs))) < 1e-10
    assert np.all(p1[3:] > prices[3:])
