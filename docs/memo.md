# Memo: likely price effects of a merger between Firm 3 and Firm 4

**From:** Xiangzhong (Shaun) Meng · **Date:** 10 November 2026

*Illustrative exercise. The merger is hypothetical, and the data are a semi-fabricated teaching dataset of breakfast-cereal sales (Nevo, 2000), in which firms are anonymous. The figures show how the analysis works; they say nothing about any real company. Each figure comes from a file in `results/` of this repository.*

## The question

Firm 3 and Firm 4 sell ready-to-eat cereals in 94 local markets (47 cities, two quarters each). Firm 3 has two brands and Firm 4 has three. Would a merger between them lead to materially higher prices, and how large would cost savings need to be to prevent that?

## Short answer

The merger would probably raise the parties' prices a little: by about 2–3% on average under the two more flexible demand models, and by under 2% under the simplest one. Rivals would barely change their prices. The loss to consumers is about 0.35–0.6% of what they spend on cereal. A saving of around 5% in the merged firm's marginal costs would be enough to offset the price rise. The average hides large local effects: in individual cities, single brands rise by up to 13–21%, depending on the model. If this were a real case, those cities would deserve a closer look.

## How the analysis was done

1. **Shares and concentration.** We measured each firm's share of cereal sales in every city and the change in market concentration (the HHI) caused by the merger. We compared these with the European Commission's screening thresholds. The CMA uses no fixed thresholds.
2. **Demand.** We estimated how strongly shoppers respond to price, and where they go when a brand becomes dearer, with three standard models. The simplest (logit) assumes that shoppers switch in proportion to brands' market shares. The second (nested logit) lets "soggy" and "crunchy" cereals form two groups whose members compete more closely. The third (random coefficients, the approach of Berry, Levinsohn and Pakes) lets tastes and price sensitivity vary across shoppers, including with income.
3. **Simulation.** Assuming each firm sets the prices that maximise its profit given its rivals' prices, we inferred each brand's cost from today's prices. We then recomputed the prices the merged firm and its rivals would set.

## Results

**Screening.** The parties' combined share averages 21% across the cities. It ranges from 2% to 59%, and it is 25% or more in 36 of 94 cities. Concentration is already high: the HHI averages 3,408, and the merger adds 236 points on average. Under the Commission's guidelines the merger falls inside the concentration safe harbour in 47 cities and outside it in the other 47. In 11 of those 47 the combined share is still below the 25% the guidelines treat as an indication of no concern (`concentration_summary.csv`, `eu_hmg_classification.csv`). That outcome calls for a closer look, but is not a finding of harm.

**Closeness of competition.** The two firms are not each other's closest rivals. When a merging brand raises its price, 3–12% of the customers it loses go to the other party, depending on the brand and the model. The rest go to the other three firms' brands or stop buying cereal (`merging_party_demand.csv`). This limited closeness is the main reason the price effect is small.

**Simulated price effects** (`merger_summary.csv`):

| | Simple logit | Nested logit | Random coefficients |
|---|---|---|---|
| Merging brands, average price change | +1.7% | +2.6% | +2.7% |
| Rival brands, average price change | +0.04% | +0.16% | +0.12% |
| Consumer loss, % of cereal spending | 0.35% | 0.60% | 0.58% |
| Cost saving that would offset the price rise | 3.4% | 5.5% | 4.8% |

The two more flexible models agree closely. The simple logit gives a smaller effect, because it assumes that many lost customers stop buying cereal altogether rather than switching to the other party.

**Comparison.** The same tools applied to a merger of the two largest firms (Firms 1 and 2, with a combined share of about 75%) give price rises of 7–15% and a consumer loss of 4.5–9.4% of spending. The cost savings needed to offset them would be implausibly large. The Firm 3 + Firm 4 merger is of a different order.

## How robust is this?

We varied the main assumptions one at a time (`sensitivity.csv`):

- **Closeness within the soggy and crunchy groups** (the nesting parameter, which the data cannot pin down): the merging brands' price rise moves from 2.2% to 2.8%.
- **Price sensitivity** (the 95% confidence range of the simple logit): 1.6% to 1.9%.
- **Market size** (how many potential cereal buyers there are): 1.1% to 2.9%. This assumption matters about as much as the choice of model.
- **Cost savings:** a 2.5% saving roughly halves the price rise under the two flexible models and almost removes it under logit. A 5% saving turns it into a small price fall under all three.

Across these checks the average price rise stays between about 1% and 3%.

## Caveats

- **The data are teaching data**, built to resemble real scanner data, and the firms are anonymous. A real case would need the parties' own sales, margins and switching evidence (surveys, internal documents, win/loss data).
- **The model is static.** It ignores entry, repositioning of brands, retailer bargaining and any coordination between firms. Costs are inferred from the assumption that today's prices are profit-maximising. For a handful of Firm 1 and Firm 2 brands in some cities that assumption implies costs of zero or less, which is a warning sign about it there.
- **Market size and the nesting structure are assumptions**, and they move the results.
- **Cost savings count only if they can be verified.** They must also be specific to the merger and likely to be passed on to customers. The 5% figure is a benchmark for the parties to meet, not an estimate of savings they will achieve.
- **Local effects are larger than the averages.** A real review would focus on the cities where the parties are strongest.
