# Project 3 requirement checklist

| Requirement | Deliverable / command |
|---|---|
| Full Unit 2 baseline runtime | `docs/BASELINE.md`, `results/performance/baseline_full.json`; `make baseline-full` |
| Runtime profile | `profile_baseline.txt`, `profile_optimized.txt`, binary `.prof` files; `make profile` |
| Complexity analysis | `docs/BASELINE.md`, `complexity.csv`, `complexity.png`; `make complexity` |
| Numerical warnings/convergence | `stability_summary.csv`, replicate-level `stability.csv`, `stability_events.json`; `make stability-check` |
| At least two optimization categories | Algorithmic improvements, array programming, numerical stability, parallelization in `src/optimized.py` and `src/runner.py` |
| Before/after code and trade-offs | `docs/OPTIMIZATION.md` |
| Timing visualization | `results/performance/timing_comparison.png` and `complexity.png` |
| Correctness regression | `tests/test_regression.py`, `full_regression.json`; `make test` |
| Required Makefile targets | `profile`, `complexity`, `benchmark`, `parallel`, `stability-check` |

The optional extensions were not attempted. All recorded timing values are measured,
not extrapolated. They depend on hardware and software versions.
