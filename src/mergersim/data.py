"""Loaders for the two teaching datasets that ship with pyblp 1.3.0.

Neither dataset is downloaded: both CSV files are installed with the pyblp package, so the
scripts run without a network connection once pyblp is installed.

* Cereal (`load_cereal`): the "fake cereal" product data of Nevo (2000a), "A practitioner's
  guide to estimation of random-coefficients logit models of demand", taken by pyblp from
  Nevo's Matlab code as archived on Eric Rasmusen's website. That archive describes them as
  semi-fabricated: "motivated by real scanner data, but it is not real and should not be used
  to make any inference". Firms, brands and cities are anonymous IDs, not real companies.
  There are 94 markets (47 cities x 2 quarters) and the same 24 brands in each market.
  The file includes Nevo's 20 pre-computed excluded instruments for price
  (`demand_instruments0` ...). The agent file holds 20 simulated consumers per market with
  demeaned demographics (income, income squared, age, child) drawn from the CPS.
* Autos (`load_autos`): the automobile data of Berry, Levinsohn and Pakes (1995), as extracted
  by Andrews, Gentzkow and Shapiro (2017) from the original GAUSS code of BLP (1999).
  There are 20 annual US markets (1971-1990), 2,217 model-years and 26 firms. The excluded
  instruments are BLP's "sums of characteristics" instruments, built by pyblp.
"""

from __future__ import annotations

import pandas as pd
from pyblp import data as _pyblp_data

CEREAL_INSTRUMENTS = [f"demand_instruments{i}" for i in range(20)]
AUTO_INSTRUMENTS = [f"demand_instruments{i}" for i in range(8)]


def _add_outside_share(products: pd.DataFrame) -> pd.DataFrame:
    """Add the outside good's share s_0 = 1 - sum of inside shares in each market."""
    out = products.copy()
    out["outside_share"] = 1.0 - out.groupby("market_ids")["shares"].transform("sum")
    return out


def load_cereal() -> pd.DataFrame:
    """Nevo's (2000a) fake cereal product data: one row per brand and city-quarter market."""
    products = pd.read_csv(_pyblp_data.NEVO_PRODUCTS_LOCATION)
    return _add_outside_share(products)


def load_cereal_agents() -> pd.DataFrame:
    """Nevo's simulated consumers: 20 per market, with Monte Carlo draws and demographics."""
    return pd.read_csv(_pyblp_data.NEVO_AGENTS_LOCATION)


def load_autos() -> pd.DataFrame:
    """BLP's (1995) automobile product data: one row per model and year.

    `car_ids` is copied to `product_ids` so that both datasets share the same column names.
    """
    products = pd.read_csv(_pyblp_data.BLP_PRODUCTS_LOCATION)
    products["product_ids"] = products["car_ids"]
    return _add_outside_share(products)
