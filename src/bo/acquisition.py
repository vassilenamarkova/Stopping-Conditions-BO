"""Acquisition function + optimizer: decides which point is most worth evaluating next, given the current GP's beliefs."""
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

def propose_next_point(model, train_y, bounds, num_restarts=10, raw_samples=64):
    """Return the single next point worth evaluating, chosen by maximizing Expected Improvement over the GP's current posterior.
    """
    acqf = LogExpectedImprovement(model, best_f=train_y.max())

    next_x, acq_value = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return next_x