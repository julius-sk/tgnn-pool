#!/usr/bin/env bash
# Everything that needs no pool, no GPU and no dataset.
# Writes one JSON record per piece into results/.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### B1  snapshot isolation (exhaustive over interleavings)"
python derive/b1_check.py

echo
echo "### B3  snapshot capacity by layout"
python derive/b3_capacity.py

echo
echo "### B2  placement crossover, both layouts"
python derive/b2_crossover.py --layout delta
python derive/b2_crossover.py --layout shared_rowptr --no-write

echo
echo "### X1  leakage, mock backend"
python exp/x1_leakage.py

echo
echo "### plans for the cluster half"
python exp/x2b_contention.py --dry-run
python exp/x2_admission.py --dry-run
python exp/x3_rolling_origin.py --dry-run

echo
echo "records written to results/"
ls -1 results/*.json 2>/dev/null | tail -8 || true
