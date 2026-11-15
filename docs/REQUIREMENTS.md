# Project 3 checklist

This table shows where each part of the project can be found. The performance results are saved under `results/performance/`.

| Project requirement | Files or command |
|---|---|
| Runtime of the full original simulation | `docs/BASELINE.md`, `baseline_full.json`; `make baseline-full` |
| Profiling and main bottlenecks | `profile_baseline.txt`, `profile_optimized.txt`; `make profile` |
| Complexity analysis | `docs/BASELINE.md`, `complexity.csv`, `complexity.png`; `make complexity` |
| Warnings and convergence checks | `stability_summary.csv`, `stability.csv`, `stability_events.json`; `make stability-check` |
| At least two types of optimization | Algorithmic changes, array calculations, numerical stability and parallel execution in `src/optimized.py` and `src/runner.py` |
| Explanation of the changes, before/after code and limitations | `docs/OPTIMIZATION.md` |
| Performance comparison plot | `timing_comparison.png` and `complexity.png`; `make benchmark` and `make complexity` |
| Tests comparing the results | `tests/test_regression.py`, `full_regression.json`; `make test` |
| Parallel simulation | `make parallel WORKERS=2` |

All reported times were measured by running the code. They were not estimated by scaling up a smaller run. Times will depend on the computer and package versions used. The optional extensions were not attempted.
