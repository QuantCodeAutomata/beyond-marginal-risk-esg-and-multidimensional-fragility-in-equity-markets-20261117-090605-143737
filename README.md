# Beyond Marginal Risk: ESG and Multidimensional Fragility in Equity Markets

Replication of selected results from *Beyond Marginal Risk: ESG and
Multidimensional Fragility in Equity Markets*.

This repository implements the paper's fixed-effects panel methodology
(Equations 3-7), its multiple-testing and bootstrap inference procedures
(Holm adjustment, Rademacher wild-cluster bootstrap), and six of its
experiments:

1. Primary marginal ESG × Down tests with full risk adjustment (return,
   volatility, liquidity channels).
2. Conventional risk attenuation of the Joint80 ESG association.
3. Joint80 model, exact state decomposition (PairRV80 = RVOnly + Joint80),
   and construct validations (sector×month FE, Amihud liquidity leg).
4. Paired standardized IAS vs. WAKIS ESG score architecture comparison.
5. Fragility ladder (1D/2D/3D tail indicators) and threshold-sensitivity
   diagnostics (J70-J90).
6. Entry/persistence dynamics and compositional decomposition (pillar
   scores, within-firm vs. firm-mean ESG).

## Data

Price history (the monthly S&P 500 market-state series `Down_t`/`Up_t`,
Eq. 4) is loaded through the QuantX Dataloader REST API, using the SPY ETF
as a tracking proxy for the S&P 500 price index because Massive index
symbols require a separate entitlement (see `data/data_loader.py`).

The paper's firm-level inputs — MSCI monthly ESG snapshots (Industry
Adjusted Score, Weighted Average Key Issue Score, E/S/G pillar scores) and
Compustat annual accounting variables — are commercially licensed datasets
with no permitted REST source in this environment (not Dataloader, not
Massive, not Yahoo Finance). `data/generate_synthetic.py` generates a
firm-month panel with the paper's reported sample dimensions (526 firms,
83 months) and ESG descriptive statistics (IAS/WAKIS correlation, update
frequencies, pooled SD) so that every regression, Holm correction, and
bootstrap routine can be exercised end to end. Every experiment's output
records this provenance explicitly; see `results/RESULTS.md` for the full
list of deviations and what could not be reproduced on real data.

## Repository layout

```
src/        Shared library: config, panel construction, FE regressions,
            Holm/wild-bootstrap inference, decomposition, reporting.
exp/        One script per experiment (exp1.py .. exp6.py).
data/       data_loader.py (QuantX Dataloader) and generate_synthetic.py.
tests/      Unit tests (synthetic fixtures) and a market-data smoke test.
results/    Every table, log, and RESULTS.md produced by the experiments.
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic fixtures
python -m pytest                        # full suite, needs API keys for the market-data test
```

## Running the experiments

Each experiment can be run at the reduced, time-limited configuration used
for the results in this repository, or at the paper's full configuration
with `--full` (526 firms, 9,999 wild-bootstrap replications):

```bash
python -m exp.exp1   # primary marginal ESG x Down tests
python -m exp.exp2   # conventional risk attenuation of Joint80
python -m exp.exp3   # Joint80 model, decomposition, construct validation
python -m exp.exp4   # IAS vs WAKIS architecture comparison
python -m exp.exp5   # fragility ladder and threshold grid
python -m exp.exp6   # entry/persistence and pillar/within-firm decomposition

# Paper-scale configuration (526 firms, 9,999 bootstrap replications;
# slower, see results/RESULTS.md for the timing estimate):
python -m exp.exp1 --full
python -m exp.exp2 --full
python -m exp.exp3 --full
python -m exp.exp4 --full
python -m exp.exp5 --full
python -m exp.exp6 --full
```

`reproduce.sh` runs the test suite and all six experiments in sequence.

## Results

See `results/RESULTS.md` for every metric next to the paper's reported
value, a reproduction verdict for each, and the full list of deviations
from the paper and of results not reproduced.
