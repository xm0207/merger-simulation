"""The modelling choices used by the scripts, written down in one place.

Scripts import these instead of repeating them, so the README, the memo and the code agree on
what "the logit model" or "the merger" means.
"""

from __future__ import annotations

import pandas as pd

from .data import CEREAL_INSTRUMENTS
from .estimation import FittedDemand, estimate_logit, estimate_nested_logit, estimate_rc_nevo
from .merger import Merger

# The hypothetical merger studied, and a much larger one used only as a point of comparison.
# Firm IDs are the anonymous IDs of Nevo's fake cereal data.
MAIN_MERGER = Merger("Firm 3 + Firm 4", (3, 4))
CONTRAST_MERGER = Merger("Firm 1 + Firm 2", (1, 2))

# Nests for the nested logit: the only product grouping in the data is the `mushy` dummy
# (cereals that go soggy in milk versus those that stay crunchy).
NEST_COLUMN = "mushy"

# The nesting parameter is not identified in these data once brand fixed effects are included
# (unconstrained 2SLS gives rho > 1), so it is fixed. 0.5 is a central judgement, not an
# estimate; 0.25 and 0.75 are reported as sensitivities.
CENTRAL_RHO = 0.5
RHO_GRID = (0.25, 0.5, 0.75)


def cereal_logit(products: pd.DataFrame) -> FittedDemand:
    """Plain logit, brand fixed effects, price instrumented with Nevo's 20 instruments."""
    return estimate_logit(products, instruments=CEREAL_INSTRUMENTS, label="Logit (2SLS)")


def cereal_nested_logit(products: pd.DataFrame, rho: float = CENTRAL_RHO) -> FittedDemand:
    """Nested logit (mushy / not mushy), brand fixed effects, rho fixed, alpha by 2SLS given rho."""
    return estimate_nested_logit(products, NEST_COLUMN, CEREAL_INSTRUMENTS, rho=rho,
                                 label=f"Nested logit (rho = {rho:g})")


def central_models(products: pd.DataFrame, agents: pd.DataFrame) -> list[FittedDemand]:
    """The three demand specifications compared throughout: logit, nested logit, BLP."""
    return [cereal_logit(products), cereal_nested_logit(products),
            estimate_rc_nevo(products, agents, label="Random coefficients (BLP)")]
