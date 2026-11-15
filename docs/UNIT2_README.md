# Unit 2 simulation: background

This is a short description of the original project. For the updated commands and Project 3 results, see the main `README.md`, `BASELINE.md` and `OPTIMIZATION.md`.

The Unit 2 study is based on Freedman (2008), *Randomization Does Not Justify Logistic Regression*. It compares the logistic regression treatment coefficient, the plug-in estimate and the ITT estimate when the logistic model is misspecified.

There are two DGPs: the Freedman example and a nonlinear extension. Treatment is randomized in both cases. Thus, the nonlinear example has model misspecification, not treatment confounding.

## Original files

- `src/dgps.py` generates the potential outcomes and treatment assignments.
- `src/methods.py` calculates the true effect and the three estimates.
- `src/simulation.py` runs the original simulation.
- `src/metrics.py` creates summary tables and plots.
- `tests/tests.py` contains the original tests.
- `ADEMP.md` describes the simulation design.
- `data/` contains simulation results, and `results/` contains summaries and plots.

Copies of the four original source files are kept in `baseline/`. Some of the copies in `src/` are still used by the new code and tests, so they have been retained. The updated Makefile uses `src/runner.py` for the optimized simulation.
