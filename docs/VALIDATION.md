# Final validation

- All 18 tests passed (`results/performance/test_results.txt`).
- All 8,000 paired baseline/optimized replicates agree within 1e-7.
- Full serial and two-worker optimized CSVs are exactly identical after CSV parsing.
- Final data files each contain 4,000 rows and match the benchmark output.
- All five preserved source files match the upload byte-for-byte; SHA-256 hashes
  are recorded in `results/performance/baseline_source_sha256.json`.
- The required profiling, complexity, benchmark, parallel and stability targets
  ran successfully, as did the full simulation/metrics pipeline.
- Performance charts and rendered scientific PDFs were visually inspected.

One test-session warning comes from the host's unrelated pytest/anyio plugin
(deprecated pytest API). It is not a simulation numerical warning or test failure.

The current environment's dependency versions are in `requirements-tested.txt`;
`requirements.txt` lists the minimal direct dependencies.
