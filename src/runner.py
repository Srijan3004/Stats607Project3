"""Reproducible scenario-level parallel simulation, identical data across engines."""
from concurrent.futures import ProcessPoolExecutor
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dgps import generate_freedman_population, generate_nonlinear_population, randomize_treatment
from methods import calc_true_delta, estimate_effects as baseline_effects
from optimized import estimate_effects as optimized_effects

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234567891
NS = [100, 500, 1000, 5000]


def scenario(task):
    dgp, n, reps, seed, engine = task
    # Stable by (DGP,n,rep), independent of scheduling, n ordering and workers.
    seq = np.random.SeedSequence([seed, int(dgp == 'nonlinear'), n])
    seeds = seq.generate_state(reps+1)
    gen = generate_freedman_population if dgp == 'freedman' else generate_nonlinear_population
    Z, yc, yt = gen(n, seed=int(seeds[0]))
    truth = calc_true_delta(yc, yt)
    estimator = baseline_effects if engine == 'baseline' else optimized_effects
    rows = []
    for rep in range(reps):
        X, y = randomize_treatment(yc, yt, seed=int(seeds[rep+1]))
        b, p, i = estimator(X, Z, y)
        rows.append((dgp, n, rep, truth, b, p, i))
    return rows


def run(reps=1000, workers=1, engine='optimized', ns=None, seed=SEED):
    if reps < 1 or workers < 1 or engine not in ('baseline', 'optimized'):
        raise ValueError('Positive reps/workers and a known engine required')
    ns = NS if ns is None else list(ns)
    if any(n < 2 for n in ns):
        raise ValueError('n must be at least two')
    tasks = [(d,n,reps,seed,engine) for d in ('freedman','nonlinear') for n in ns]
    if workers == 1:
        groups = list(map(scenario, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            groups = list(pool.map(scenario, tasks))
    return pd.DataFrame([r for g in groups for r in g], columns=['dgp','n','rep','True_Delta','MLE_Beta2','Plugin_Delta','ITT_Delta'])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replicates', type=int, default=1000)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    df = run(a.replicates, a.workers)
    (ROOT/'data').mkdir(exist_ok=True)
    for d, group in df.groupby('dgp'):
        group.drop(columns='dgp').to_csv(ROOT/'data'/f'sim_results_{d}.csv', index=False)
    print(f'Saved {len(df)} replicates using {a.workers} worker(s)')
