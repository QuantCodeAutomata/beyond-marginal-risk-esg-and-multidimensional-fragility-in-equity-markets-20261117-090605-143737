#!/usr/bin/env bash
# Runs the test suite and every experiment at the reduced, time-limited
# configuration described in results/RESULTS.md. Pass --full to run the
# paper's full configuration (526 firms, 9,999 wild-bootstrap replications)
# for every experiment instead.
set -uo pipefail

FULL_FLAG=""
if [[ "${1:-}" == "--full" ]]; then
  FULL_FLAG="--full"
fi

python -m pytest -m "not market_data"

for exp in exp1 exp2 exp3 exp4 exp5 exp6; do
  echo "=== Running exp.${exp} ${FULL_FLAG} ==="
  python -m "exp.${exp}" ${FULL_FLAG}
done
