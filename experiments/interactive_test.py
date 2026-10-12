from bo.objective import sine, SINE_BOUNDS
from bo.interactive import build_interactive_plot

build_interactive_plot(sine, SINE_BOUNDS, n_init=4, n_iters=10,
                        outpath="outputs/sine_interactive.html")