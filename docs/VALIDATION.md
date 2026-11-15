# Checks on the completed project

The following checks were completed in the environment used for the reported results:

- All 18 tests passed. The output is in `results/performance/test_results.txt`.
- All 8,000 paired baseline and optimized results agreed within a tolerance of 1e-7.
- The serial and two-worker optimized results were exactly equal after reading the CSV files.
- Each final DGP data file contained 4,000 rows and matched the benchmark output.
- The four source files in `baseline/` matched the original uploaded files. The recorded file hashes are in `results/performance/baseline_source_sha256.json`.
- The simulation, summary plots and required Makefile commands completed successfully.
- The performance plots and simulation plots were checked for readability.

The test output includes one warning from a pytest plugin about a deprecated Python interface. It is not a numerical warning from the simulation and did not cause a test to fail.

The required packages are listed in `requirements.txt`. The versions used for the recorded measurements are described in `results/performance/environment.json`. The tests and timing commands can be run again on another computer; its runtime results may differ.
