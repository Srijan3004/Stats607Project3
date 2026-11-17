# Baseline performance

## Simulation setup

This project continues my Unit 2 simulation study. There are two DGPs, Freedman and nonlinear, with population sizes 100, 500, 1,000 and 5,000. For each DGP and population size, the population is held fixed and treatment is randomized 1,000 times. Thus, the full study has 8,000 replicates.

The original code is kept in `baseline/` so that the changes can be compared with it.

## Runtime of the original code

The full original simulation took **14.749 seconds** in the environment used for these measurements.

| DGP | Replicates | Time (seconds) |
|---|---:|---:|
| Freedman | 4,000 | 7.435 |
| Nonlinear | 4,000 | 7.309 |
| Total, including time between the two runs | 8,000 | 14.749 |

This measurement includes generating data, fitting the models and saving the CSV files. It does not include importing Python packages or making the summary plots. Only the output folder was changed when running the original simulation. The timing results are saved in `results/performance/baseline_full.json`.

All three estimators returned finite values in this run. However, the original code suppresses fitting warnings, so this alone does not establish that every fit converged. The separate warning check below looks at this more carefully.

There was also a problem with the random seed. Although the simulation sets a seed at the beginning, the data-generation and treatment-assignment functions call `np.random.seed(None)` when no seed is supplied. Therefore, running the original program again does not reproduce the same data.

## Where does the code spend time?

The profiler was run on a smaller study with 400 fits: 100 replicates for each DGP at n = 1,000 and n = 5,000.

| Part of the original code | Number of calls | Time including functions called inside it (seconds) |
|---|---:|---:|
| Whole smaller study | 1 | 1.125 |
| `estimate_effects` | 400 | 1.093 |
| `Logit.fit` | 400 | 0.498 |
| `add_constant` | 1,200 | 0.329 |
| Creating the Logit model | 400 | 0.174 |

These times overlap, so they should not be added. About 97% of the measured time is spent inside `estimate_effects`. Apart from fitting the model, repeatedly creating the design matrices also takes a noticeable amount of time.

The full profiler tables are in `results/performance/profile_baseline.txt` and `profile_optimized.txt`. The optimized version took 1.093 seconds under profiling, compared with 1.125 seconds for the original version. Profiling itself adds some time, so the separate benchmark is used to compare actual runtimes.

## Computational complexity

There are only three regression coefficients. Generating data and assigning treatment both take O(n) time. Each Newton iteration also takes O(n) time when the number of coefficients is fixed. Thus, with k iterations, fitting one model takes approximately O(kn) time. The predictions and ITT calculation add another O(n) amount of work.

The optimized method has the same order of complexity. It reduces some repeated calculations, but it does not change the linear dependence on n for a fixed number of iterations. Near separation, more iterations may be needed. Checking smaller Newton steps also adds some work.

For R replicates and several population sizes, the total work is approximately O(R sum_j k_j n_j). The arrays used for one fit require O(n) memory. Using several processes requires more memory because each process works on its own data.

To examine this empirically, batches of 30 replicates were timed for n = 100, 250, 500, 1,000, 2,500, 5,000, 10,000 and 20,000. Each setting was repeated five times. The batch time was divided by 30, and a line was fitted to log median time against log n.

| DGP | Version | Estimated slope |
|---|---|---:|
| Freedman | Original | 0.498 |
| Freedman | Optimized | 0.501 |
| Nonlinear | Original | 0.492 |
| Nonlinear | Optimized | 0.487 |

![Runtime against population size](../results/performance/complexity.png)

The dashed lines show the fitted log-log relationships. The slopes are close to 0.5 over this range, but this does not mean that the algorithms have O(sqrt(n)) complexity. For smaller n, setting up each fit takes a substantial part of the total time. The number of iterations can also change with the data. The theoretical cost is still O(n) when the iteration count is fixed.

## Numerical warnings and convergence

There were 100 replicates for each DGP at n = 20, 100, 500, 1,000 and 5,000, giving 1,000 fits per version. The population seed was 908 and the treatment seeds were 0 to 99. Five additional datasets were used to check difficult cases.

At the four population sizes from the original study, there were **no warnings, failed fits or nonfinite estimates in the 800 checked fits per version**.

The smaller size n = 20 gave more problems:

| DGP | Original nonconverged fits | Original errors | Warning events | Fits flagged by optimized code |
|---|---:|---:|---:|---:|
| Freedman | 11 | 1 | 11 | 12 |
| Nonlinear | 11 | 1 | 40 | 12 |

Each row is based on 100 replicates. A fit can give several warnings, so the warning count need not equal the number of failed fits. The ITT estimate was undefined in 12 Freedman replicates and 11 nonlinear replicates.

In the extra checks, the original method returned finite but nonconverged estimates for constant outcomes and perfect separation. These produced 19 and 40 warnings, respectively. The optimized method marked these fits as failed. An empty treatment arm and a rank-deficient design also led to failures. Both versions completed the large-covariate check without warnings.

The optimized method gave no numerical warnings in these checks, but it still reported failed fits. It returns missing MLE and plug-in estimates when a fit is not reliable, instead of treating a large nonconverged coefficient as a valid estimate. The detailed results are in `stability_summary.csv`, `stability.csv` and `stability_events.json` under `results/performance/`.

## Timing comparison

For the comparison below, both versions used the same populations and treatment assignments. Each full study was run three times, alternating the order of the versions. The table gives the median and range.

| Version | Median time (seconds) | Range (seconds) | Speedup over original serial version |
|---|---:|---:|---:|
| Original, serial | 11.612 | 11.553-11.631 | 1.00x |
| Optimized, serial | 10.300 | 10.296-10.875 | 1.13x |
| Optimized, two workers | 6.667 | 6.388-7.105 | 1.74x |

![Comparison of full-study runtimes](../results/performance/timing_comparison.png)

These timings include generating data and collecting the results, but exclude saving CSV files. The parallel timing includes starting the worker processes. This comparison also uses explicit seeds and a different simulation driver. Therefore, the original serial time here is not the same measurement as the 14.749-second run above.

These measurements used Linux, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3 and statsmodels 0.15.0. Numerical-library threads were limited to one per process. The environment reported nine logical CPUs, but these were not necessarily available exclusively to this job. Times may differ on another computer. See `environment.json` and `benchmark.csv` for the saved details.

The commands are `make baseline-full`, `make profile`, `make complexity`, `make benchmark` and `make stability-check`. The raw `.prof` files, logs and duplicate simulation CSVs are generated locally and are ignored by Git; the main tables and plots are kept in the repository.
