# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this implements

A replication of selected results from *Beyond Marginal Risk: ESG and
Multidimensional Fragility in Equity Markets*: the paper's fixed-effects
panel methodology (Equations 3-7), its Holm multiple-testing adjustment,
its Rademacher wild-cluster-bootstrap inference, and six of its experiments
(primary marginal ESG × Down tests, Joint80 risk attenuation, Joint80
decomposition and construct validation, IAS vs. WAKIS architecture
comparison, the fragility ladder and threshold grid, and entry/persistence
and pillar/within-firm decomposition). See `README.md` for the experiment
list and `results/RESULTS.md` for results, deviations and what is not
reproduced.

## Layout

```
src/        Shared library: config.py (all parameters), panel construction
            (data_construction.py), FE regressions (regression.py),
            Holm/wild-bootstrap inference (inference.py), state
            decomposition (decomposition.py), the end-to-end panel
            assembly (panel_pipeline.py, pipeline.py), and table writers
            (reporting.py).
exp/        One script per experiment, exp1.py .. exp6.py. Each has a
            `main(full: bool = False)` entry point and can be run as
            `python -m exp.exp<N>` or `python -m exp.exp<N> --full`.
data/       data_loader.py (QuantX Dataloader REST market data) and
            generate_synthetic.py (synthetic ESG/accounting panel, used
            because MSCI ESG and Compustat data have no permitted REST
            source; see results/RESULTS.md).
tests/      pytest tests: unit tests on synthetic fixtures, plus one
            market-data smoke test guarded by the `market_data` marker.
results/    Every table, JSON log and plot produced by the experiments,
            plus results/RESULTS.md (paper comparison and deviations).
```

## Conventions

- All experiment and model parameters live in `src/config.py` as
  module-level `UPPER_CASE` variables, each with a comment naming its unit
  and its source in the paper (or "not specified in the paper"). Do not
  hardcode parameters inside `src/`, `exp/`, or `data/`.
- Market data is loaded only through `data/data_loader.py`, which calls the
  QuantX Dataloader REST API (`https://quantx-api.limex.pro/apis/dataloader`)
  for OHLCV bars. Synthetic data is generated only through
  `data/generate_synthetic.py`. No other module fetches or generates data.
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` file via `python-dotenv`; `.env.example` holds only
  empty placeholders. Never hardcode or log a key.
- Downloaded market data and generated synthetic panels are never committed;
  only the tables, logs and plots written to `results/` are.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, synthetic fixtures only
python -m pytest                        # full suite (needs API keys for the market-data test)

python -m exp.exp1            # reduced, time-limited configuration
python -m exp.exp1 --full     # paper's configuration (526 firms, 9,999 wild-bootstrap reps)
# exp2 .. exp6 accept the same --full flag

bash reproduce.sh             # runs the tests and all six experiments
```

## Notes for future changes

- After rerunning any experiment command, it overwrites the corresponding
  files in `results/`. The numbers, tables and verdicts written by hand in
  `results/RESULTS.md` must then be checked against the new files and
  updated manually — `results/RESULTS.md` is not regenerated automatically.
- `src/panel_pipeline.py` and `src/pipeline.py` build the one shared panel
  (point-in-time ESG, lagged accounting controls, channel-specific ex ante
  exposure blocks, Down/Up market state) that every experiment script
  consumes; keep new experiments on this same shared pipeline instead of
  writing a private panel builder or a private regression call.
- The wild-cluster bootstrap and leave-one-month-out loops are the slowest
  steps; `config.py` keeps a `_PAPER` and a `_RUN` variant of every
  size-scaling parameter (firm count, bootstrap replications) so that
  `--full` can select the paper's configuration without changing the
  estimating equations.
