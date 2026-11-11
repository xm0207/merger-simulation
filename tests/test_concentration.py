import numpy as np
import pandas as pd
import pytest

from mergersim import Merger, ShareTable, delta_hhi, eu_hmg_indication, hhi


def test_hhi_by_hand():
    assert hhi([50, 30, 20]) == pytest.approx(2500 + 900 + 400)
    assert hhi([100]) == 10_000
    assert hhi([10] * 10) == pytest.approx(1000)


def test_delta_hhi_is_2ab():
    assert delta_hhi(30, 20) == pytest.approx(1200)
    # the same number from the definition: HHI after minus HHI before
    assert hhi([50, 50]) - hhi([30, 20, 50]) == pytest.approx(delta_hhi(30, 20))


@pytest.mark.parametrize("hhi_post, delta, combined, max_share, expected", [
    (900, 300, 30, 20, "HHI < 1,000"),
    (1800, 200, 30, 20, "HHI 1,000-2,000 and delta < 250"),
    (1800, 300, 30, 20, "outside safe harbours"),
    (3000, 100, 30, 20, "HHI > 2,000 and delta < 150"),
    (3000, 236, 21, 11, "combined share < 25%"),
    (3000, 100, 55, 52, "party share >= 50%"),
])
def test_eu_hmg_bands(hhi_post, delta, combined, max_share, expected):
    assert expected in eu_hmg_indication(hhi_post, delta, combined, max_share)


@pytest.fixture
def toy_sales():
    # two markets; market X shares 40/30/20/10 by volume, market Y has prices that change value shares
    return pd.DataFrame({
        "market_ids": ["X"] * 4 + ["Y"] * 3,
        "firm_ids": [1, 2, 3, 4, 1, 2, 3],
        "product_ids": ["a", "b", "c", "d", "a", "b", "c"],
        "shares": [0.20, 0.15, 0.10, 0.05, 0.1, 0.1, 0.2],
        "prices": [1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0],
    })


def test_share_table_volume_and_value(toy_sales):
    table = ShareTable(toy_sales)
    vol = table.firm_shares("volume").set_index(["market_ids", "firm_ids"])["share_pct"]
    assert vol[("X", 1)] == pytest.approx(40)
    assert vol[("Y", 3)] == pytest.approx(50)
    val = table.firm_shares("value").set_index(["market_ids", "firm_ids"])["share_pct"]
    assert val[("Y", 1)] == pytest.approx(50)  # 0.3 of the 0.6 spent


def test_sql_concentration_matches_arithmetic(toy_sales):
    out = ShareTable(toy_sales).concentration(Merger("2+3", (2, 3))).set_index("market_ids")
    x = out.loc["X"]
    assert x["hhi_pre"] == pytest.approx(hhi([40, 30, 20, 10]))
    assert x["hhi_post"] == pytest.approx(hhi([40, 50, 10]))
    assert x["delta_hhi"] == pytest.approx(delta_hhi(30, 20))
    assert x["combined_share_pct"] == pytest.approx(50)
    assert x["max_party_share_pct"] == pytest.approx(30)
    assert np.allclose(out["hhi_post"] - out["hhi_pre"], out["delta_hhi"])
