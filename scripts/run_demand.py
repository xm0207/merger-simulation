"""Estimate demand: logit (OLS and IV), nested logit and random-coefficients logit (pyblp).

Cereal (Nevo's fake data) carries the merger analysis. The BLP automobile data are used only
for the OLS-versus-IV comparison, where the textbook price-endogeneity problem is starkest.

Writes to results/: demand_estimates.csv, nested_logit_identification.csv, elasticities_summary.csv
and merging_party_demand.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mergersim import demand_diagnostics, estimate_logit, nested_logit_regression
from mergersim.data import AUTO_INSTRUMENTS, CEREAL_INSTRUMENTS, load_autos, load_cereal, load_cereal_agents
from mergersim.estimation import NEVO_PI0, NEVO_SIGMA0
from mergersim.specs import MAIN_MERGER, NEST_COLUMN, RHO_GRID, central_models, cereal_nested_logit

AUTO_CHARACTERISTICS = ["hpwt", "air", "mpd", "space"]


def iv_rows(dataset: str, model: str, fit_or_result, keep: list[str]) -> list[dict]:
    result = getattr(fit_or_result, "result", fit_or_result)
    rows = []
    for name in keep:
        if name in result.coef.index:
            rows.append({"dataset": dataset, "model": model, "method": result.method, "parameter": name,
                         "coef": result.coef[name], "se": result.se[name], "nobs": result.nobs,
                         "first_stage_f": result.first_stage_f.get(name, np.nan)})
    return rows


def rc_rows(fit) -> list[dict]:
    r = fit.results
    rows = [{"dataset": "cereal", "model": fit.label, "method": "GMM (pyblp, 1-step)", "parameter": "prices",
             "coef": float(r.beta[0, 0]), "se": float(r.beta_se[0, 0]), "nobs": r.problem.N}]
    for i, row in enumerate(r.sigma_labels):
        for j, col in enumerate(r.sigma_labels):
            if NEVO_SIGMA0[i, j] != 0:
                rows.append({"dataset": "cereal", "model": fit.label, "method": "GMM (pyblp, 1-step)",
                             "parameter": f"sigma[{row}]", "coef": float(r.sigma[i, j]), "se": float(r.sigma_se[i, j]),
                             "nobs": r.problem.N})
        for j, col in enumerate(r.pi_labels):
            if NEVO_PI0[i, j] != 0:
                rows.append({"dataset": "cereal", "model": fit.label, "method": "GMM (pyblp, 1-step)",
                             "parameter": f"pi[{row} x {col}]", "coef": float(r.pi[i, j]), "se": float(r.pi_se[i, j]),
                             "nobs": r.problem.N})
    return rows


def summarise(diag: pd.DataFrame, dataset: str) -> dict:
    own = diag["own_elasticity"]
    return {
        "dataset": dataset, "model": diag["model"].iloc[0], "product_markets": len(diag),
        "own_elasticity_median": own.median(), "own_elasticity_p25": own.quantile(0.25),
        "own_elasticity_p75": own.quantile(0.75), "inelastic_products": int((own > -1).sum()),
        "cross_elasticity_median": diag["cross_elasticity_median"].median(),
        "diversion_to_outside_median_pct": 100 * diag["diversion_to_outside"].median(),
        "margin_median_pct": diag["margin_pct"].median(),
        "nonpositive_costs": int((diag["cost"] <= 0).sum()),
        "nonpositive_costs_by_firm": "; ".join(f"firm {k}: {v}" for k, v in
                                              diag.loc[diag["cost"] <= 0, "firm_ids"].value_counts().sort_index().items()),
    }


def nest_rival_averages(products: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """For each product, the average of `columns` over the other products in its nest and market.

    Rivals' cost shifters move a product's within-nest share but not its own unobserved quality,
    so they are candidate instruments for ln(s_{j|g}); unlike product counts, they vary across markets.
    """
    g = products.groupby(["market_ids", NEST_COLUMN])
    return pd.DataFrame({f"nest_rival_{c}": (g[c].transform("sum") - products[c]) / (g[c].transform("size") - 1)
                         for c in columns})


def nested_logit_identification(cereal: pd.DataFrame) -> pd.DataFrame:
    """Unconstrained nested-logit 2SLS under alternative specifications: how stable is rho?"""
    rivals = nest_rival_averages(cereal, CEREAL_INSTRUMENTS)
    data = pd.concat([cereal, rivals], axis=1)
    specs = [
        ("brand FE", "product_ids", [], CEREAL_INSTRUMENTS, "Nevo's 20"),
        ("brand FE", "product_ids", [], CEREAL_INSTRUMENTS + list(rivals.columns), "Nevo's 20 + their nest-rival averages"),
        ("brand FE", "product_ids", [], list(rivals.columns), "nest-rival averages only"),
        ("constant only", None, [], CEREAL_INSTRUMENTS, "Nevo's 20"),
        ("constant, sugar, mushy", None, ["sugar", "mushy"], CEREAL_INSTRUMENTS, "Nevo's 20"),
    ]
    rows = []
    for controls, fe, chars, instruments, label in specs:
        r = nested_logit_regression(data, NEST_COLUMN, instruments, chars, fe)
        rows.append({"controls": controls, "instruments": label, "rho": r.coef["log_within_share"],
                     "rho_se": r.se["log_within_share"], "price_coef": r.coef["prices"], "price_se": r.se["prices"],
                     "first_stage_f_price": r.first_stage_f["prices"],
                     "first_stage_f_within_share": r.first_stage_f["log_within_share"],
                     "admissible": bool(0 <= r.coef["log_within_share"] < 1 and r.coef["prices"] < 0)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    cereal, agents = load_cereal(), load_cereal_agents()
    autos = load_autos()
    estimates, summaries = [], []

    # 1. Plain logit on cereal: OLS and IV, with and without brand fixed effects
    cereal_logits = {}
    for fe in (None, "product_ids"):
        chars = ["sugar", "mushy"] if fe is None else []
        for ins in (None, CEREAL_INSTRUMENTS):
            fit = estimate_logit(cereal, chars, fixed_effects=fe, instruments=ins)
            name = f"Logit, {'brand FE' if fe else 'no FE'}"
            cereal_logits[(fe, ins is not None)] = fit
            estimates += iv_rows("cereal", name, fit, ["prices", "sugar", "mushy", "const"])
            fit.label = f"{name} ({fit.result.method})"
            summaries.append(summarise(demand_diagnostics(fit), "cereal"))

    # 2. Nested logit: the unconstrained 2SLS estimate of rho, then rho fixed on a grid
    unconstrained = nested_logit_regression(cereal, NEST_COLUMN, CEREAL_INSTRUMENTS)
    estimates += iv_rows("cereal", "Nested logit, brand FE, rho estimated (inadmissible)", unconstrained,
                         ["prices", "log_within_share"])
    ident = nested_logit_identification(cereal)
    ident.to_csv(out / "nested_logit_identification.csv", index=False, float_format="%.4g")
    print(ident.to_string(index=False))
    for rho in RHO_GRID:
        fit = cereal_nested_logit(cereal, rho)
        estimates += iv_rows("cereal", f"Nested logit, brand FE, rho fixed at {rho:g}", fit, ["prices"])
        summaries.append(summarise(demand_diagnostics(fit), "cereal"))

    # 3. The three central models; the random-coefficients one is estimated here by pyblp
    models = central_models(cereal, agents)
    rc = models[-1]
    estimates += rc_rows(rc)
    rc_diag = demand_diagnostics(rc)
    summaries.append(summarise(rc_diag, "cereal"))

    # 4. Autos: OLS versus IV, the classic demonstration (BLP instruments)
    for ins in (None, AUTO_INSTRUMENTS):
        fit = estimate_logit(autos, AUTO_CHARACTERISTICS, fixed_effects=None, instruments=ins)
        estimates += iv_rows("autos", "Logit, characteristics", fit, ["prices", "const"] + AUTO_CHARACTERISTICS)
        fit.label = f"Logit, characteristics ({fit.result.method})"
        summaries.append(summarise(demand_diagnostics(fit), "autos"))

    est = pd.DataFrame(estimates)
    est.to_csv(out / "demand_estimates.csv", index=False, float_format="%.6g")
    print(est.to_string(index=False))
    summ = pd.DataFrame(summaries)
    summ.to_csv(out / "elasticities_summary.csv", index=False, float_format="%.4g")
    print(summ.to_string(index=False))

    # 5. The merging parties' products under the three central models: own elasticity, diversion
    #    to the partner and to the outside good, and the implied margin (averaged over markets)
    rows = []
    for fit in models:
        for market_id in fit.market_ids:
            idx = fit.rows(market_id)
            firms = fit.products["firm_ids"].to_numpy()[idx]
            demand = fit.market(market_id)
            D = demand.diversion()
            own = np.diag(demand.elasticities())
            for j in np.flatnonzero(MAIN_MERGER.involves(firms)):
                partner = MAIN_MERGER.involves(firms) & (firms != firms[j])
                rows.append({"model": fit.label, "product_ids": fit.products["product_ids"].to_numpy()[idx][j],
                             "firm_ids": firms[j], "own_elasticity": own[j],
                             "diversion_to_partner_pct": 100 * D[j, partner].sum(),
                             "diversion_to_outside_pct": 100 * (1 - D[j].sum())})
    party = pd.DataFrame(rows).groupby(["model", "firm_ids", "product_ids"], sort=False).mean().reset_index()
    margins = pd.concat([demand_diagnostics(f) for f in models[:2]] + [rc_diag])
    margins = margins[MAIN_MERGER.involves(margins["firm_ids"])].groupby(["model", "product_ids"])["margin_pct"].mean()
    party = party.merge(margins.reset_index(), on=["model", "product_ids"])
    party.to_csv(out / "merging_party_demand.csv", index=False, float_format="%.3f")
    print(party.to_string(index=False))


if __name__ == "__main__":
    main()
