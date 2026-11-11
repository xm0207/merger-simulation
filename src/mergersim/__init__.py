"""Merger screening and merger simulation with logit, nested logit and random-coefficients demand."""

from .concentration import ShareTable, delta_hhi, eu_hmg_indication, hhi
from .demand import DemandModel, LogitDemand, NestedLogitDemand, RandomCoefficientsDemand
from .equilibrium import BertrandNash, compensating_cost_reductions
from .estimation import (FittedDemand, IVResult, LogitFit, NestedLogitFit, RCFit, berry_inversion,
                         estimate_logit, estimate_nested_logit, estimate_rc_nevo, nested_logit_regression,
                         two_sls, within_nest_shares)
from .merger import Merger, ownership_matrix
from .screening import (critical_elasticity, critical_loss, diversion_matrix, guppi, guppi_vector, partner_mask,
                        share_diversion, upp)
from .simulation import MergerSimulator, SimulationResult, demand_diagnostics, hypothetical_monopolist_test

__all__ = [
    "BertrandNash",
    "DemandModel",
    "FittedDemand",
    "IVResult",
    "LogitDemand",
    "LogitFit",
    "Merger",
    "MergerSimulator",
    "NestedLogitDemand",
    "NestedLogitFit",
    "RCFit",
    "RandomCoefficientsDemand",
    "ShareTable",
    "SimulationResult",
    "berry_inversion",
    "compensating_cost_reductions",
    "critical_elasticity",
    "critical_loss",
    "delta_hhi",
    "demand_diagnostics",
    "diversion_matrix",
    "estimate_logit",
    "estimate_nested_logit",
    "estimate_rc_nevo",
    "eu_hmg_indication",
    "guppi",
    "guppi_vector",
    "hhi",
    "hypothetical_monopolist_test",
    "nested_logit_regression",
    "ownership_matrix",
    "partner_mask",
    "share_diversion",
    "two_sls",
    "upp",
    "within_nest_shares",
]
