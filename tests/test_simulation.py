import numpy as np
import pytest

from mergersim import BertrandNash, Merger, MergerSimulator, critical_loss, hypothetical_monopolist_test
from mergersim.estimation import LogitFit


class FixedLogitFit(LogitFit):
    """A LogitFit with a chosen alpha, so the simulation can be tested without estimation."""

    def __init__(self, products, alpha):
        super().__init__("toy logit", products, alpha, result=None)


@pytest.fixture
def fit(products):
    return FixedLogitFit(products, alpha=-2.0)


def test_simulation_outputs_are_consistent(fit):
    result = MergerSimulator(fit, Merger("1+2", (1, 2))).run()
    p = result.products
    assert len(p) == 15 and len(result.markets) == 3
    assert (p.loc[p["merging"], "price_change_pct"] > 0).all()
    assert (result.markets["cs_change"] < 0).all()
    assert result.markets["foc_residual_pre"].max() < 1e-12
    summary = result.summary()
    assert summary["merging_price_change_pct"] > summary["rivals_price_change_pct"] > 0


def test_cost_reduction_equal_to_cmcr_leaves_prices_unchanged(fit):
    """Lowering each merging product's cost by its reported CMCR must leave prices where they were."""
    sim = MergerSimulator(fit, Merger("1+2", (1, 2)))
    products, _ = sim.run_market("M0")
    cmcr = products.loc[products["merging"], "cmcr_pct"].to_numpy() / 100
    demand = fit.market("M0")
    firms = products["firm_ids"].to_numpy()
    costs = products["cost"].to_numpy()
    c_new = costs.copy()
    c_new[products["merging"].to_numpy()] *= 1 - cmcr
    p1 = BertrandNash(demand, Merger("1+2", (1, 2)).ownership(firms)).solve(c_new)
    assert np.allclose(p1, demand.prices0, atol=1e-9)


def test_efficiencies_reduce_price_effects(fit):
    sim = MergerSimulator(fit, Merger("1+2", (1, 2)))
    no_eff = sim.run().summary()["merging_price_change_pct"]
    eff = sim.run(cost_reduction=0.05).summary()["merging_price_change_pct"]
    assert eff < no_eff


def test_hypothetical_monopolist_test(fit):
    hmt = hypothetical_monopolist_test(fit, ssnip=0.05)
    assert len(hmt) == 3
    assert np.allclose(hmt["critical_loss"], [critical_loss(0.05, m) for m in hmt["margin"]])
    assert (hmt["profitable"] == (hmt["profit_change_pct"] > 0)).all()
    assert ((hmt["actual_loss"] > 0) & (hmt["actual_loss"] < 1)).all()
