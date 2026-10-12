# Stopping Conditions for BO

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the demos

```bash
python experiments/objective_test.py      # objective function correctness checks
python experiments/objective_plot.py      # plots of all three raw objective functions
python experiments/gp_test.py             # GP fitting: posterior mean/variance vs. true values
python experiments/gp_plot_test.py        # single GP state plot (mean + uncertainty band)
python experiments/acquisition_test.py    # acquisition function suggests a next point
python experiments/loop_test.py           # full BO loop, printed + plotted per iteration
python experiments/interactive_test.py    # interactive visualization, all three functions
python experiments/convergence_test.py    # convergence plots, all three functions
```

Each saves its outputs (PNGs, interactive HTML) to `outputs/`.

## Running tests

```bash
pytest -v
```

## Structure

- `src/bo/objective.py` — objective functions BO optimises over: `sine` (hand-written), `x_squared` (hand-written, negated), `levy` (BoTorch's built-in 1D synthetic benchmark, negated)
- `src/bo/gp.py` — GP surrogate fitting (`fit_gp`), with input/output normalisation
- `src/bo/acquisition.py` — Expected Improvement acquisition function + optimizer (`propose_next_point`)
- `src/bo/loop.py` — the core BO loop(`run_bo_loop`)
- `src/bo/plotting.py` — diagnostic plots: GP state (mean + uncertainty band), convergence curve
- `src/bo/interactive.py` — interactive  visualization through BO iterations
- `experiments/` — scripts that run a specific check or demo
- `tests/` — pytests

