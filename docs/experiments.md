# The numbered pieces

Six pieces. Three run anywhere; three need a host attached to the pool. Each
section says what the piece decides, what it costs, what it depends on, and
what would make it fail.

B1 has its own file: `docs/B1-snapshot-isolation.md`.

---

## B2 — placement crossover

`python derive/b2_crossover.py [--scale full|window] [--layout ...]`

**Decides:** at how many hosts pooled memory stops winning, and which resource
runs out first for each architecture.

**Why it has to exist:** the pool is dual-headed. No experiment on this cluster
can reach three hosts. Any claim about more than two therefore comes from a
model calibrated on the points that can be measured, or it does not get made.

**What each architecture runs out of:**

| | per-host residency | binding resource |
|---|---|---|
| pool | the working set only | pool capacity (constant in N), pool bandwidth (divided by N) |
| partitioned host memory | `S · r(N) / N` | host DRAM, at the *load* peak rather than the steady size |
| GPU memory | `S / N` | aggregate VRAM |

The load peak is the part that is easy to miss: holding a partition and loading
one are different sizes, measured here at 1.85×. An architecture that fits in
steady state and not at load does not fit.

**Two inputs are not measured, and the script refuses to invent them:**

- `alpha`, the halo growth rate in `r(N) = 1 + α(N−1)`. Every partitioned-memory
  figure depends on it. Measure it by partitioning one graph into 1, 2 and 4
  parts and recording resident bytes.
- pool aggregate bandwidth. Until X2b runs, the script reports the two ends of
  the range instead of a number, and they are far apart: saturation at about 14
  hosts, or at about 2.

**It would fail if** the working-set assumption breaks — it holds while hosts
only read, and stops the moment one writes embeddings back.

---

## B3 — snapshot capacity, and what the layout costs

`python derive/b3_capacity.py [--pool-bytes N] [--also-current]`

**Decides:** how many snapshots fit, and which of the three structure layouts
is responsible.

The feature table does not grow with the number of snapshots. The structure
does, and how fast is a layout choice:

| layout | structure cost | grows with T? |
|---|---|---|
| `expanded` | edges copied into every later snapshot, `≈ E(T+1)/2` | linearly, large constant |
| `delta` | column indices once; row pointers once **per snapshot** | linearly, constant `(N+1)·8` |
| `shared_rowptr` | row pointers once, column indices once, one timestamp per edge | not at all |

`delta` is what the project runs. `shared_rowptr` needs a sampler that can cut
inside a row, which the sampler in use cannot do — so the difference between
those two rows is the price of a missing sampler feature, in bytes.

**The model is checked before it is used.** It reproduces the measured
row-pointer size to within 0.3%. A capacity argument built on a model that
cannot reproduce a measured size is not worth making.

**Result at 500 GB, 120 snapshots, 768-dim fp32 features (374 GB):**

| layout | structure | total | fits | max snapshots |
|---|---|---|---|---|
| expanded | 745.0 GB | 1119.0 GB | no | 19 |
| delta | 127.3 GB | 501.3 GB | **no, by 1.3 GB** | 118 |
| shared_rowptr | 12.7 GB | 386.7 GB | yes | > 1200 |

The layout in use misses the full graph by two snapshots. That is a sharper
statement than a ratio, and it points at a specific thing to fix.

**It would fail if** allocator overhead is large: the heap expands in fixed
steps, so bytes taken exceed bytes live, and every maximum here is an upper
bound.

---

## X1 — does a sampler read the future?

`python exp/x1_leakage.py [--backend mock|pool]`

**Decides:** whether B1's argument describes what actually happens when a
sampler and an ingest run at the same time.

**Why both B1 and X1:** B1 can be right about a protocol the hardware does not
implement. X1 on its own cannot distinguish a safe design from a lucky
schedule. Neither is sufficient.

**Design note that matters:** a snapshot is modelled as a handful of large
arrays, not one object per graph row, because that is how the pool stores it. A
reader's chance of landing inside a publication in flight is roughly one over
the object count, so per-row objects would make tearing look rare for a reason
that has nothing to do with the design.

**Result (mock backend, 10 trials):** 10.7% of edges leaked under read
committed, 0% under version filtering, 9 torn reads under no filtering at all —
the shape B1 predicts.

**Open:** the pool backend. `tgnnpool/store.py` declares the interface and
refuses to guess the implementation; filling it in is the first cluster task,
and it must publish in the same four passes or the two halves are testing
different things.

**It would fail if** the absolute leak rate were quoted as a property of the
workload. It is a property of the pacing. Only the zero against the non-zero is
a claim.

---

## X2 — cost of admitting one more snapshot

`python exp/x2_admission.py --dry-run`

**Decides:** the fourth row of the three-way comparison — what each
architecture does when the graph grows.

- **pool:** publish the new arrays, flip their state. Proportional to the new
  edges. Training does not stop, and B1 says what readers see meanwhile.
  Predicted at 0.6 s for one window snapshot, from the measured ingest rate.
- **partitioned host memory:** repartition and reload. Proportional to the
  whole graph. Training stops throughout.
- **GPU memory:** does not arise — the dataset does not fit to begin with.

**The asymmetry is the finding, not a flaw.** One side keeps training because
it can; the other stops because it must.

**Blocked on one number:** nobody has timed a repartition-and-reload on this
cluster. Until that exists the ratio does not exist either. The script will not
estimate it.

---

## X2b — does a second reader halve the bandwidth?

`python exp/x2b_contention.py --dry-run`

**Decides:** whether the measured single-reader bandwidth is a per-reader rate
or a device limit that readers split.

**Cheapest unrun experiment here, and it gates the most.** Two outcomes:

- independent → saturation at roughly 27 hosts; pooling scales for this
  workload.
- shared → saturation at roughly 14 hosts on the training-level demand, and
  closer to 2 on the micro-benchmark duty cycle.

Both are publishable. The second is more interesting, and finding out after
writing the scaling argument would be expensive.

**Procedure:** one reader for a fixed interval, then two simultaneously, one
per host, same access shape as training — random rows, measured row width. A
sequential scan would report a bandwidth this workload never sees. Bind to the
NUMA node the pool is near on both hosts, or the number describes the
interconnect topology instead.

---

## X3 — what does the usual evaluation protocol overstate?

`python exp/x3_rolling_origin.py --dry-run`

**Decides:** how much accuracy is inflated by evaluating a temporal model with
a random split instead of a rolling origin.

A temporal model should be scored the way it is used: fit up to `t`, predict
`t+1`, advance. Most snapshot work does not, and the reason is cost — each
origin has needed its own partition, and partitioning is hours.

**Over a versioned pool an origin is a read filter, not a partition.** Fold
setup costs nothing, so the protocol that was unaffordable becomes the default.
That is what connects a methodological point to the storage design, and why it
belongs in the same work as B1 and X1.

**Arms:** one random split, and four rolling origins, same model, same seeds,
same epoch budget per fold. Estimated 5.7 h and 1.4 h respectively at the
measured epoch time.

**The check that matters:** fold setup must take seconds. If it does not, the
versioning is not being used and the protocol argument has nothing behind it.

**Report per-origin, not only the mean.** Early origins train on less data and
score worse for that reason alone; a mean confounds protocol with training-set
size.

**A small difference is still an answer**, and a more useful one than not
knowing.
