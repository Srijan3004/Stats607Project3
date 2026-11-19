# STAT 607 Project 3: High-Performance Simulation Study

This project continues my Unit 2 simulation study on logistic model misspecification. The aim is to reduce the runtime and improve numerical stability while keeping the statistical setup unchanged.

The study uses two data-generating mechanisms and four population sizes: 100, 500, 1,000 and 5,000. For each setting, the population is held fixed and treatment is randomized 1,000 times, giving 8,000 replicates in total.

The baseline results are discussed in `docs/BASELINE.md`, and the changes are explained in `docs/OPTIMIZATION.md`. A checklist of the project requirements is in `docs/REQUIREMENTS.md`.

## Running the project

From the project directory, first create and activate a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run:

```bash
make test       # run the tests
make            # run the optimized simulation and create summaries and plots
make evidence   # run all performance and stability checks
```

The checks can also be run separately:

```bash
make baseline-full           # time the original Unit 2 simulation
make profile                 # check where the code spends time
make complexity              # examine runtime for different population sizes
make benchmark               # compare original and optimized runtimes
make stability-check         # check warnings and convergence problems
make parallel WORKERS=2      # run the optimized simulation with two workers
```

For a quick toy trial, use:

```bash
make simulate REPLICATES=10
```

This replaces the main simulation data with the smaller run. Run `make` again to regenerate the full study.

The Makefile limits numerical-library threads to one per process so that parallel workers do not each start additional threads.

## Project files

- `baseline/`: copies of the four original Unit 2 source files.
- `src/dgps.py`: generates the populations and treatment assignments.
- `src/methods.py`: contains the original estimation functions.
- `src/simulation.py`: contains the original simulation procedure.
- `src/metrics.py`: creates summary tables and plots.
- `src/optimized.py`: contains the modified estimation method and numerical checks.
- `src/runner.py`: runs the simulation with explicit random seeds and optional parallel execution.
- `scripts/performance.py`: runs the timing, profiling, complexity and stability checks.
- `tests/`: checks the simulation and compares the original and optimized results.
- `data/`: contains the simulation results.
- `results/raw/` and `results/figures/`: contain summary tables and plots.
- `results/performance/`: contains the performance results and diagnostic summaries. Some intermediate files are generated locally but ignored by Git.
- `docs/`: explains the baseline performance, changes and validation checks.
- `ADEMP.md` and `ANALYSIS.md`: contain the original Unit 2 discussion.
