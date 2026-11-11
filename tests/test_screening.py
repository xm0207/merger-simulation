import numpy as np
import pytest

from mergersim.screening import (critical_elasticity, critical_loss, diversion_matrix, guppi, guppi_vector,
                                 partner_mask, share_diversion, upp)


def test_critical_loss_formula():
    assert critical_loss(0.05, 0.45) == pytest.approx(0.10)
    assert critical_loss(0.10, 0.40) == pytest.approx(0.20)
    assert critical_elasticity(0.05, 0.45) == pytest.approx(2.0)


@pytest.mark.parametrize("x, m", [(0.05, 0.3), (0.10, 0.6), (0.05, 0.8)])
def test_critical_loss_is_break_even(x, m):
    """Losing exactly the critical loss leaves profit unchanged: (p(1+X) - c)(1 - CL) = p - c."""
    p, q = 1.0, 1.0
    c = p * (1 - m)
    cl = critical_loss(x, m)
    assert (p * (1 + x) - c) * q * (1 - cl) == pytest.approx((p - c) * q)


def test_critical_loss_rejects_bad_inputs():
    with pytest.raises(ValueError):
        critical_loss(0.0, 0.4)


def test_diversion_from_logit_jacobian_is_share_proportional(logit):
    s = logit.shares(logit.prices0)
    D = diversion_matrix(logit.jacobian(logit.prices0))
    expected = s[None, :] / (1 - s)[:, None]
    np.fill_diagonal(expected, 0)
    assert np.allclose(D, expected)
    # what is not diverted to other products goes to the outside good
    assert np.allclose(1 - D.sum(axis=1), (1 - s.sum()) / (1 - s))


def test_share_diversion_with_stated_outside_loss():
    D = share_diversion([0.5, 0.3, 0.2], outside_diversion=0.25)
    assert D[0, 1] == pytest.approx(0.75 * 0.3 / 0.5)
    assert np.allclose(D.sum(axis=1), 0.75)


def test_upp_and_guppi_by_hand():
    # D = 0.2, partner price 10 and cost 6, own cost 5, 2% efficiency on own cost
    assert upp(0.2, 10.0, 6.0, 5.0, 0.02) == pytest.approx(0.2 * 4 - 0.1)
    # partner margin 40%, partner price 10, own price 8: 0.2 * 0.4 * 10 / 8
    assert guppi(0.2, 0.4, 10.0, 8.0) == pytest.approx(0.1)
    # GUPPI is UPP without efficiencies as a share of own price
    assert guppi(0.2, 0.4, 10.0, 8.0) == pytest.approx(upp(0.2, 10.0, 6.0, 5.0) / 8.0)


def test_guppi_vector_matches_scalar_formula():
    prices = np.array([8.0, 10.0, 5.0])
    costs = np.array([5.0, 6.0, 3.0])
    D = np.array([[0, 0.2, 0.1], [0.15, 0, 0.1], [0.1, 0.1, 0]])
    mask = partner_mask([1, 2, 3], merging_firms=[1, 2])
    g = guppi_vector(D, prices, costs, mask)
    assert g[0] == pytest.approx(guppi(0.2, 0.4, 10.0, 8.0))
    assert g[1] == pytest.approx(guppi(0.15, 3 / 8, 8.0, 10.0))
    assert g[2] == 0.0  # firm 3 is not merging
