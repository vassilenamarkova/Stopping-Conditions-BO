from bo.objective import sine, SINE_BOUNDS, x_squared, XSQUARED_BOUNDS, levy, LEVY_BOUNDS
from bo.interactive import build_interactive_plot

build_interactive_plot(sine, SINE_BOUNDS, n_init=4, n_iters=20,
                        outpath="outputs/sine_interactive.html")


build_interactive_plot(x_squared, XSQUARED_BOUNDS, n_init=4, n_iters=20,
                        outpath="outputs/x_squared_interactive.html")

build_interactive_plot(levy, LEVY_BOUNDS, n_init=4, n_iters=20,
                        outpath="outputs/levy_interactive.html")