from bo.objective import (
    sine, SINE_BOUNDS,
    x_squared, XSQUARED_BOUNDS,
    levy, LEVY_BOUNDS, LEVY_OPTIMAL_VALUE,
)
from bo.loop import run_bo_loop
from bo.plotting import plot_convergence

_, _, best_sine = run_bo_loop(sine, SINE_BOUNDS, n_init=4, n_iters=10)
plot_convergence(best_sine, true_optimum=1.0, function_name="sine",
                  outpath="outputs/convergence_sine.png")

_, _, best_xsq = run_bo_loop(x_squared, XSQUARED_BOUNDS, n_init=4, n_iters=10)
plot_convergence(best_xsq, true_optimum=0.0, function_name="x_squared",
                  outpath="outputs/convergence_x_squared.png")

_, _, best_levy = run_bo_loop(levy, LEVY_BOUNDS, n_init=4, n_iters=10)
plot_convergence(best_levy, true_optimum=LEVY_OPTIMAL_VALUE, function_name="levy",
                  outpath="outputs/convergence_levy.png")