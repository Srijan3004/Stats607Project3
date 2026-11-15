PYTHON ?= python3
WORKERS ?= 2
REPLICATES ?= 1000
export PYTHONPATH := src:$(PYTHONPATH)
# Prevent BLAS oversubscription, including in child processes.
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export MPLBACKEND := Agg
.PHONY: all simulate metrics test profile complexity benchmark parallel stability-check baseline-full evidence clean
all: simulate metrics
simulate:
	$(PYTHON) src/runner.py --replicates $(REPLICATES)
metrics:
	$(PYTHON) src/metrics.py
test:
	$(PYTHON) -m pytest tests/tests.py tests/test_regression.py -v
profile complexity benchmark stability-check baseline-full:
	$(PYTHON) scripts/performance.py $@
parallel:
	$(PYTHON) src/runner.py --replicates $(REPLICATES) --workers $(WORKERS)
evidence: baseline-full profile complexity benchmark stability-check
clean:
	$(PYTHON) -c "import pathlib,shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
