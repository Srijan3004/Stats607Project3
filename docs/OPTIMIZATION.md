# Optimization of the logistic-misspecification simulation

## Scope and statistical target

The estimands, DGPs, treatment probability 0.75, population sizes, and 1,000
randomizations per population remain unchanged. The original Unit 2 sources are
preserved in `baseline/`. The new implementation uses unpenalized maximum
likelihood; it does not replace the MLE with ridge regression, clipping, or a
pseudocount estimator. It retains the original three outputs and CSV schema.

A fresh population is drawn once per DGP/population-size combination and reused
for all treatment assignments. Vectorizing independent population draws across
replicates would change that design and is deliberately avoided.

## 1. Specialized optimization algorithm (algorithmic improvement)

**Problem.** The profile attributes most runtime to estimation: statsmodels
model construction, repeated constant-column checks, the Newton optimizer,
and general-purpose fitting infrastructure. This study needs only three
coefficients and fitted probabilities; it does not use covariance estimates.

**Before** (`baseline/methods.py`):

```python
exog = sm.add_constant(np.column_stack((X, Z)))
model = sm.Logit(Y_obs, exog).fit(disp=0, method='newton')
beta_hat_2 = model.params[1]
```

**After** (`src/optimized.py`, simplified):

```python
A = np.column_stack((np.ones(len(X)), X, z))
p = expit(A @ beta)
w = p * expit(-(A @ beta))
H = A.T @ (w[:, None] * A)
g = A.T @ (p - y)
step = np.linalg.solve(H, g)
beta = beta - scale * step
```

For negative log likelihood
`L(beta) = sum(log(1 + exp(A_i beta)) - y_i A_i beta)`, the gradient is
`A.T (p-y)` and Hessian is `A.T diag(p(1-p)) A`. The Hessian is positive
semidefinite, and a full-rank nonseparated fit has a unique finite optimum.
A linear solve replaces an explicit inverse; no covariance matrix or rich model
object is constructed. Convergence requires both a small parameter step and a
small normalized score. A backtracking step checks that loss does not increase.

**Trade-offs.** Maintaining a dedicated solver is more work than calling
statsmodels. It supports this binary three-column model only, not arbitrary
formulas, weights, offsets, standard errors, or robust inference. Regression
tests against statsmodels are therefore essential. The extra stability checks
also cost time, and speedups need not grow with n.

## 2. Remove redundant counterfactual design matrices (array programming)

**Problem.** The original repeats constant detection and allocates two n-by-3
counterfactual arrays in every replicate.

**Before:**

```python
exog_T = sm.add_constant(np.column_stack((np.ones_like(X), Z)), has_constant='add')
exog_C = sm.add_constant(np.column_stack((np.zeros_like(X), Z)), has_constant='add')
p_T = model.predict(exog_T)
p_C = model.predict(exog_C)
```

**After:**

```python
eta_c = beta[0] + beta[2] * z
eta_t = eta_c + beta[1]
```

The direct array formulas give the same linear predictors without scanning or
allocating two design matrices. Broadcasting `w[:, None] * A` computes the
Hessian without constructing an n-by-n diagonal matrix. This is an O(n)
implementation for fixed parameter dimension. Two n-by-3 float64 matrices
would occupy 48n bytes (240,000 bytes at n=5,000); this particular allocation is
removed, but this is not a measured net peak-memory saving because the new
solver has its own arrays. Original code already vectorized many operations;
there was no reason to replace every remaining simulation loop.

## 3. Stable probabilities, standardized covariate, explicit failures

**Problem.** The original hides every fitting warning and catches every
exception, does not check the convergence flag, and forms odds before taking
logs. Near separation it can return large unconverged coefficients. Very large
covariate scales can make the Hessian ill conditioned.

**Before:**

```python
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model = sm.Logit(Y_obs, exog).fit(disp=0, method='newton')
plugin_delta = np.log(alpha_T_tilde/(1-alpha_T_tilde)) - np.log(alpha_C_tilde/(1-alpha_C_tilde))
```

**After:**

```python
z = (Z - Z.mean()) / Z.std()
loss = np.sum(np.logaddexp(0, eta) - y * eta)
w = expit(eta) * expit(-eta)
log_odds = (logsumexp(-np.logaddexp(0, -eta))
            - logsumexp(-np.logaddexp(0, eta)))
```

The last expression is exactly `log(mean(p)) - log(mean(1-p))`: the two
`log(n)` terms cancel. It avoids rounding a probability to one before computing
its complement and avoids exponentiating large positive predictors.
Standardizing Z only reparameterizes the intercept/covariate coefficient;
the treatment coefficient and counterfactual predictions are invariant.

The solver reports `singular_design`, `constant_outcome`, `ill_conditioned`,
`line_search_failed`, or `nonconverged` when appropriate. Failed MLE and plug-in
values are NaN; the ITT value remains available if its two arm proportions lie
strictly between 0 and 1. The default return keeps the original tuple API;
`diagnostics=True` additionally returns status and iteration information.
A condition-number threshold of 1e12 and a 60-iteration cap are safeguards,
not mathematical proofs of separation. Extremely difficult but finite fits
can be rejected; the failure log makes this visible.

`make stability-check` instruments a copy of the original source to expose its
suppressed warnings, swallowed exceptions, and convergence flag. It changes
only observation of failures, not its fitting arithmetic. Raw messages and
conditions are in `stability_events.json`; aggregate counts are in
`stability_summary.csv`. Repeated warnings within a fit count as multiple
warning events, so event counts can exceed replicate counts.

## 4. Reproducible scenario-level processes (parallelization)

**Problem.** The original nests serial loops and calls `np.random.seed(None)`
in each DGP/assignment call. That erases the intended top-level seed.

**Before:**

```python
np.random.seed(SEED)
Z, Y_C, Y_T = generate_freedman_population(n)  # internally seed(None)
X, Y_obs = randomize_treatment(Y_C, Y_T)      # internally seed(None)
```

**After** (`src/runner.py`):

```python
seq = np.random.SeedSequence([seed, int(dgp == 'nonlinear'), n])
seeds = seq.generate_state(reps + 1)
Z, yc, yt = gen(n, seed=int(seeds[0]))
x, y = randomize_treatment(yc, yt, seed=int(seeds[rep + 1]))
with ProcessPoolExecutor(max_workers=workers) as pool:
    groups = list(pool.map(scenario, tasks))
```

Explicit seeds preserve the original random-number algorithms while making
scenario and replicate identities reproducible. Processes parallelize the
eight DGP/n scenarios and `map` preserves their order. Reproducibility does not
depend on task scheduling or worker count. BLAS uses one thread per process.

**Trade-offs.** Process startup, serialization and load imbalance limit speedup.
Only eight tasks exist, and n=5,000 tasks dominate. More than eight workers
cannot help this strategy. Processes use more memory. The implementation is
not thread-safe through its legacy global RNG calls, so it uses processes,
not threads. Serial and parallel results are tested for exact equality.

## Validation and changes in results

`tests/test_regression.py` compares both implementations on 80 identical-input
fits over both DGPs and every original n, with absolute/relative tolerance
1e-7. It also tests empty treatment arms, constant outcomes, rank deficiency,
complete separation, affine covariate rescaling, invalid data, and repeated
serial/parallel execution. The original three tests are retained.

The full benchmark compares all 8,000 paired rows and reports finite-value
agreement separately from finite/nonfinite mismatches in `full_regression.json`.
These paired checks are stronger than a two-sample distribution test here:
any discrepancy can be attributed to implementation rather than different data.
On separated/unconverged fits the optimized code intentionally returns missing
MLE/plug-in values where the original may return an unreliable finite result.
Those edge cases are not claimed to be numerically identical.

Old supplied CSVs will differ from the new reproducible runs because the Unit 2
driver reseeded from entropy. The newly timed original run is also a fresh draw.
The fair timing comparison uses the explicitly seeded common driver for both
engines; it does not compare different sets of populations.

## Measured performance and lessons

See the measured appendix below and `docs/BASELINE.md`. Profiler times include
instrumentation overhead and must not be substituted for unprofiled benchmark
wall times. The benchmark includes DGPs, estimators and DataFrame construction;
parallel timings include process startup/communication. CSV writing occurs
after the timer. All three repetitions and the environment are retained.

The main lesson is to inspect total cost: repeated model/design setup can be
material even for a small logistic model. Numerical safeguards and stable
log-probability reductions are useful but not free. Algorithm specialization
and parallelization must be assessed with unprofiled wall time and correctness
checks rather than assumed to help. We did not add GPU/JIT dependencies or
batch all fits into a large tensor; these add complexity and can substantially
increase memory for a modest study. The measured gains are for the combined
changes, not isolated causal contributions from each optimization.

## Measured appendix

| Implementation | Median (s) | Min-max (s) | Speedup vs baseline |
|---|---:|---:|---:|
| baseline, 1 worker(s) | 11.612 | 11.553-11.631 | 1.00x |
| optimized, 1 worker(s) | 10.300 | 10.296-10.875 | 1.13x |
| optimized, 2 worker(s) | 6.667 | 6.388-7.105 | 1.74x |

Serial time decreased by 11.3%; two-worker time decreased
by 42.6% relative to serial baseline. Two workers give
1.54x speedup relative to optimized serial, below ideal 2x.
For this workload the process implementation gives the larger practical gain;
the specialized serial solver provides only a modest net improvement once
its stability safeguards are included. No per-change ablation was performed.

Across 8,000 paired fits, every value is finite for both implementations and
all values match within absolute tolerance 1e-7. Maximum absolute differences:
MLE 2.22e-15, plug-in
4.11e-15, ITT
8.88e-16. There are zero finite/nonfinite
mismatches. The supplied regression suite has 18 passing tests.

The stability audit finds 24 problematic random fits across the two n=20
conditions. The optimized version flags them rather than trusting finite
unconverged coefficients. Its zero emitted warnings does not mean zero failed
fits. No convergence issue was observed in the 800 audit fits at original n.
