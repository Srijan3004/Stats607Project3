# STAT 607 Project 3: High-Performance Simulation Study

Optimizes the supplied Unit 2 logistic-misspecification study: two data-generating
mechanisms, four population sizes (100, 500, 1000, 5000), and 1,000 treatment
randomizations per fixed population, for 8,000 replicates total.

Start with **docs/BASELINE.md** and **docs/OPTIMIZATION.md**. The full checklist
is in **docs/REQUIREMENTS.md**. All measured evidence is included under
`results/performance/`.

## Reproduce

Python 3.10+ recommended. From this directory:

```bash
python3 -m pip install -r requirements.txt
make test
make all                     # full optimized simulation + original summaries/plots
make profile
make complexity
make benchmark               # 3 complete runs per engine/worker configuration
make stability-check
make parallel WORKERS=2      # full optimized study, deterministic multiprocessing
make baseline-full           # original Unit 2 simulation, output isolated
```

`make evidence` runs the five evidence-producing targets sequentially. Use
`make simulate REPLICATES=10` for a quick smoke run, but this overwrites the main
data with a small study; run `make all` to restore full results. `PYTHON=...` may
select a different interpreter. Make fixes BLAS/OpenMP threads to one for fair
comparisons and avoids oversubscription with processes.

## Files

- `baseline/`: byte-for-byte copies of all five supplied Unit 2 source modules.
- `src/dgps.py`, `src/methods.py`, `src/simulation.py`: original code retained for
  reference and regression; the default Makefile now uses `src/runner.py`.
- `src/optimized.py`: standardized, safeguarded Newton logistic MLE and stable
  plug-in probabilities, with no unnecessary covariance calculation.
- `src/runner.py`: explicit per-scenario/per-replicate seeds and process workers.
- `scripts/performance.py`: baseline timing, profiling, paired benchmarks,
  empirical complexity, and diagnostic runs.
- `data/`, `results/raw/`, `results/figures/`: full optimized study and the
  original style of scientific summaries/plots.
- `results/performance/`: raw timing/diagnostic CSVs, paired full-study estimates,
  figures, environment metadata, and profiler evidence.
- `docs/UNIT2_README.md`, `ADEMP.md`, `ANALYSIS.md`: original explanatory material.
  These describe Unit 2; consult the new documentation for measured Project 3 claims.

## Scientific interpretation and reproducibility

Treatment remains randomized in both DGPs. The nonlinear DGP has model
misspecification, not treatment confounding (some original prose used that term).
The finite population stays fixed across treatment replicates, as in Unit 2.
The original functions call `np.random.seed(None)` when the driver omits seeds,
so its top-level seed did not make the full study reproducible. The new runner
passes explicit seeds. As a result its newly generated samples need not match
old CSVs. Baseline-versus-optimized comparisons use exactly the same inputs.

The original summaries use complete-case deletion across all estimator columns;
this behavior is retained for continuity. Failure frequencies are reported
separately; do not interpret complete-case summaries as unconditional performance
when failures are common. Nonexistent/unconverged MLEs are recorded as missing
rather than replaced with penalized estimates.
