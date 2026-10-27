#!/usr/bin/env bash
# The half that needs a host attached to the pool.
#
# Run the pieces in this order: X2b first because it is minutes and unblocks
# B2, then X1 on the pool backend, then X2, then X3 which is the only one that
# costs hours.
#
# Before anything: record what the pool already holds. These experiments may
# share it with other work, and a changed baseline afterwards is the one
# failure that cannot be undone.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "This script does not run anything on its own, because each step needs a"
echo "decision that depends on the cluster. It prints the order and the"
echo "preconditions. Read docs/experiments.md, then run the steps by hand."
echo
cat <<'PLAN'
0.  Record the pool's current contents. Keep the output.

1.  X2b, contention curve                                  minutes
      host A:  python exp/x2b_contention.py --readers 1
      both:    python exp/x2b_contention.py --readers 2 --rank 0   (host A)
               python exp/x2b_contention.py --readers 2 --rank 1   (host B)
    Put the SUM into configs/machine.yaml -> pool.gather_bytes_per_s_aggregate
    Rerun: python derive/b2_crossover.py

2.  X1 on the pool backend                                 hours of coding
      Implement PoolStore in tgnnpool/store.py with the SAME four passes as
      MockStore, attaching read-only for the reader.
      python exp/x1_leakage.py --backend pool
    Expect the same shape as the mock run or stop and find out why.

3.  X2, snapshot admission                                 hours
      python exp/x2_admission.py --backend pool
      python exp/x2_admission.py --backend distdgl --partition-cmd '<cmd>'
    Put the seconds into configs/machine.yaml -> distdgl.repartition_seconds

4.  X3, rolling origin                                     ~7 h
      python exp/x3_rolling_origin.py --run --epochs 12
    Check first that fold setup takes seconds, not minutes.

5.  Record the pool's contents again and compare with step 0.
PLAN
