"""Demand estimation: logit and nested logit by linear IV on Berry's (1994) inversion, and the
random-coefficients logit by GMM in pyblp.

Berry's inversion turns the logit share equation into a linear regression. With s_0 the
outside good's share,
    ln(s_j) - ln(s_0) = delta_j = x_j beta + alpha p_j + xi_j          (logit)
    ln(s_j) - ln(s_0) = x_j beta + alpha p_j + rho ln(s_{j|g}) + xi_j  (nested logit)
Price is chosen by firms that see the unobserved quality xi_j, so it is correlated with the
error and OLS biases alpha towards zero (demand looks less elastic than it is). Two-stage least
squares with instruments that shift price but not xi corrects this.

Why classes here: each estimated specification (`LogitFit`, `NestedLogitFit`, `RCFit`) holds
its parameters for all markets and has one job, `market(market_id)`, which returns the
`DemandModel` of that market. The merger simulator only ever calls that method, so it treats
all three specifications the same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .demand import DemandModel, LogitDemand, NestedLogitDemand, RandomCoefficientsDemand


# ---------------------------------------------------------------- linear IV

@dataclass
class IVResult:
    """Coefficients, heteroskedasticity-robust (HC1) standard errors and first-stage F statistics."""

    method: str
    coef: pd.Series
    se: pd.Series
    nobs: int
    first_stage_f: dict = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.coef, "se": self.se, "t": self.coef / self.se})


def _robust_cov(Xh: np.ndarray, u: np.ndarray) -> np.ndarray:
    n, k = Xh.shape
    bread = np.linalg.pinv(Xh.T @ Xh)
    meat = (Xh * u[:, None] ** 2).T @ Xh
    return n / (n - k) * bread @ meat @ bread


def two_sls(y: pd.Series, exog: pd.DataFrame, endog: pd.DataFrame | None = None,
            instruments: pd.DataFrame | None = None) -> IVResult:
    """OLS (no `instruments`) or 2SLS of y on [exog, endog], instrumenting endog with `instruments`.

    The first-stage F for each endogenous regressor is the robust Wald statistic that its
    excluded instruments are jointly zero, divided by their number.
    """
    endog = pd.DataFrame(index=exog.index) if endog is None else endog
    X = pd.concat([exog, endog], axis=1)
    names = list(X.columns)
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    if instruments is None or endog.shape[1] == 0:
        b = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        u = yv - Xv @ b
        cov = _robust_cov(Xv, u)
        return IVResult("OLS", pd.Series(b, names), pd.Series(np.sqrt(np.diag(cov)), names), len(yv))
    Z = pd.concat([exog, instruments], axis=1).to_numpy(float)
    first = np.linalg.lstsq(Z, Xv, rcond=None)[0]
    Xh = Z @ first
    b = np.linalg.lstsq(Xh, yv, rcond=None)[0]
    u = yv - Xv @ b  # structural residuals use the actual regressors
    cov = _robust_cov(Xh, u)
    f_stats = {}
    q = instruments.shape[1]
    for i, name in enumerate(endog.columns):
        col = exog.shape[1] + i
        v = Xv[:, col] - Z @ first[:, col]
        V = _robust_cov(Z, v)[-q:, -q:]
        g = first[-q:, col]
        f_stats[name] = float(g @ np.linalg.pinv(V) @ g / q)
    return IVResult("2SLS", pd.Series(b, names), pd.Series(np.sqrt(np.diag(cov)), names), len(yv), f_stats)


# ---------------------------------------------------------------- fitted specifications

class FittedDemand(ABC):
    """A demand specification estimated on every market; hands out one `DemandModel` per market."""

    label: str

    def __init__(self, products: pd.DataFrame):
        self.products = products.reset_index(drop=True)
        self._rows = {m: np.flatnonzero(self.products["market_ids"].to_numpy() == m)
                      for m in self.products["market_ids"].unique()}

    @property
    def market_ids(self) -> list:
        return list(self._rows)

    def rows(self, market_id) -> np.ndarray:
        return self._rows[market_id]

    @abstractmethod
    def market(self, market_id) -> DemandModel:
        """The demand model of one market, at that market's observed prices."""


class LogitFit(FittedDemand):
    """Estimated plain logit. delta_j is recovered exactly from shares, so it includes xi_j."""

    def __init__(self, label: str, products: pd.DataFrame, alpha: float, result: IVResult):
        super().__init__(products)
        self.label, self.alpha, self.result = label, float(alpha), result
        self.delta = berry_inversion(self.products)

    def market(self, market_id) -> LogitDemand:
        r = self.rows(market_id)
        return LogitDemand(self.products["prices"].to_numpy()[r], self.delta[r], self.alpha)


class NestedLogitFit(FittedDemand):
    """Estimated nested logit; delta_j = ln(s_j / s_0) - rho ln(s_{j|g})."""

    def __init__(self, label: str, products: pd.DataFrame, alpha: float, rho: float,
                 nest_column: str, result: IVResult):
        super().__init__(products)
        self.label, self.alpha, self.rho, self.nest_column, self.result = label, float(alpha), float(rho), nest_column, result
        self.delta = berry_inversion(self.products) - self.rho * np.log(within_nest_shares(self.products, nest_column))

    def market(self, market_id) -> NestedLogitDemand:
        r = self.rows(market_id)
        return NestedLogitDemand(self.products["prices"].to_numpy()[r], self.delta[r], self.alpha, self.rho,
                                 self.products[self.nest_column].to_numpy()[r])


class RCFit(FittedDemand):
    """Random-coefficients logit estimated by pyblp; parameters and simulated consumers are copied
    out of pyblp's results so that `RandomCoefficientsDemand` can evaluate the model itself."""

    def __init__(self, label: str, results):
        problem = results.problem
        products = pd.DataFrame({
            "market_ids": problem.products.market_ids.ravel(),
            "firm_ids": problem.products.firm_ids.ravel(),
            "product_ids": problem.products.product_ids.ravel(),
            "prices": problem.products.prices.ravel(),
            "shares": problem.products.shares.ravel(),
        })
        super().__init__(products)
        self.label, self.results = label, results
        self.alpha = float(results.beta[list(results.beta_labels).index("prices")][0])
        self.delta = np.asarray(results.delta).ravel()
        self.x2 = np.asarray(problem.products.X2)
        self.price_column = list(results.sigma_labels).index("prices")  # columns of X2 follow sigma's labels
        self.sigma, self.pi = np.asarray(results.sigma), np.asarray(results.pi)
        agents = problem.agents
        self._agent_market = agents.market_ids.ravel()
        self._nodes, self._demographics, self._weights = agents.nodes, agents.demographics, agents.weights

    def market(self, market_id) -> RandomCoefficientsDemand:
        r = self.rows(market_id)
        a = self._agent_market == market_id
        return RandomCoefficientsDemand(self.products["prices"].to_numpy()[r], self.delta[r], self.alpha,
                                        self.x2[r], self.sigma, self.pi, self._nodes[a],
                                        self._demographics[a], self._weights[a], self.price_column)


# ---------------------------------------------------------------- estimators

def berry_inversion(products: pd.DataFrame) -> np.ndarray:
    """Mean utilities of plain logit, delta_j = ln(s_j) - ln(s_0), from observed shares."""
    return np.log(products["shares"].to_numpy()) - np.log(products["outside_share"].to_numpy())


def within_nest_shares(products: pd.DataFrame, nest_column: str) -> np.ndarray:
    """s_{j|g} = s_j / sum of shares of the products in j's nest and market."""
    nest_total = products.groupby(["market_ids", nest_column])["shares"].transform("sum")
    return (products["shares"] / nest_total).to_numpy()


def _design(products: pd.DataFrame, characteristics: Sequence[str], fixed_effects: str | None) -> pd.DataFrame:
    if fixed_effects:
        X = pd.get_dummies(products[fixed_effects], prefix="fe", dtype=float)
    else:
        X = pd.DataFrame({"const": 1.0}, index=products.index)
    for c in characteristics:
        X[c] = products[c].astype(float)
    return X


def estimate_logit(products: pd.DataFrame, characteristics: Sequence[str] = (),
                   fixed_effects: str | None = "product_ids", instruments: Sequence[str] | None = None,
                   label: str | None = None) -> LogitFit:
    """Plain logit by OLS (`instruments=None`) or 2SLS, with optional product fixed effects."""
    products = products.reset_index(drop=True)
    y = pd.Series(berry_inversion(products), name="delta")
    exog = _design(products, characteristics, fixed_effects)
    endog = products[["prices"]].astype(float)
    Z = products[list(instruments)].astype(float) if instruments else None
    result = two_sls(y, exog, endog, Z)
    label = label or f"Logit ({result.method})"
    return LogitFit(label, products, result.coef["prices"], result)


def nested_logit_regression(products: pd.DataFrame, nest_column: str, instruments: Sequence[str],
                            characteristics: Sequence[str] = (),
                            fixed_effects: str | None = "product_ids") -> IVResult:
    """The unconstrained nested-logit 2SLS regression of ln(s_j/s_0) on price and ln(s_{j|g}).

    Returned as is, whatever the value of rho, so that an inadmissible estimate can be reported.
    """
    products = products.reset_index(drop=True)
    y = pd.Series(berry_inversion(products), name="delta")
    exog = _design(products, characteristics, fixed_effects)
    endog = pd.DataFrame({"prices": products["prices"].astype(float),
                          "log_within_share": np.log(within_nest_shares(products, nest_column))})
    return two_sls(y, exog, endog, products[list(instruments)].astype(float))


def estimate_nested_logit(products: pd.DataFrame, nest_column: str, instruments: Sequence[str],
                          characteristics: Sequence[str] = (), fixed_effects: str | None = "product_ids",
                          rho: float | None = None, label: str | None = None) -> NestedLogitFit:
    """Nested logit by 2SLS.

    With `rho=None`, both price and ln(s_{j|g}) are instrumented and rho is estimated; an
    estimate outside [0, 1) is inconsistent with utility maximisation and raises ValueError.
    With `rho` given, the nesting parameter is fixed (calibrated) and only alpha and the other
    coefficients are estimated, from ln(s_j / s_0) - rho ln(s_{j|g}) on price.
    """
    products = products.reset_index(drop=True)
    y = pd.Series(berry_inversion(products), name="delta")
    log_within = np.log(within_nest_shares(products, nest_column))
    exog = _design(products, characteristics, fixed_effects)
    Z = products[list(instruments)].astype(float)
    if rho is None:
        result = nested_logit_regression(products, nest_column, instruments, characteristics, fixed_effects)
        rho = result.coef["log_within_share"]
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"estimated nesting parameter {rho:.3f} is outside [0, 1)")
        label = label or "Nested logit (2SLS)"
    else:
        result = two_sls(y - rho * log_within, exog, products[["prices"]].astype(float), Z)
        label = label or f"Nested logit (rho = {rho:g})"
    return NestedLogitFit(label, products, result.coef["prices"], rho, nest_column, result)


# Starting values of Nevo (2000a), as used in pyblp's tutorial; zeros in Pi stay fixed at zero.
NEVO_SIGMA0 = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
NEVO_PI0 = np.array([
    [5.4819, 0.0, 0.2037, 0.0],
    [15.8935, -1.2000, 0.0, 2.6342],
    [-0.2506, 0.0, 0.0511, 0.0],
    [1.2650, 0.0, -0.8091, 0.0],
])


def estimate_rc_nevo(products: pd.DataFrame, agents: pd.DataFrame, label: str = "Random coefficients (BLP)") -> RCFit:
    """Nevo's (2000a) random-coefficients specification, estimated by one-step GMM in pyblp.

    Linear part: price plus brand fixed effects. Random coefficients on a constant, price,
    sugar and mushy, with interactions with income, income squared, age and a child dummy.
    This is the specification of pyblp's Nevo tutorial.
    """
    import pyblp

    pyblp.options.verbose = False
    formulations = (pyblp.Formulation("0 + prices", absorb="C(product_ids)"),
                    pyblp.Formulation("1 + prices + sugar + mushy"))
    agent_formulation = pyblp.Formulation("0 + income + income_squared + age + child")
    cols = ["market_ids", "firm_ids", "product_ids", "shares", "prices", "sugar", "mushy"]
    cols += [c for c in products.columns if c.startswith("demand_instruments")]
    problem = pyblp.Problem(formulations, products[cols], agent_formulation, agents)
    results = problem.solve(NEVO_SIGMA0, NEVO_PI0, optimization=pyblp.Optimization("bfgs", {"gtol": 1e-5}),
                            method="1s")
    return RCFit(label, results)
