"""GP surrogate fitting.
Builds a model of the objective from the points evaluated so far."""
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll


def fit_gp(train_x, train_y, bounds):
    """
    Fit a GP to all points evaluated so far.
    bounds: (2, d) tensor, the function's real domain.
    """
    input_transform = Normalize(d=train_x.shape[-1], bounds=bounds) # d is the input dimentionality; bounds is the real domain, so it knows how to map it to [0,1] internally
    outcome_transform = Standardize(m=1)                            # single output per point

    # BoTorch GP model class
    model = SingleTaskGP(
        train_x, train_y,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    # training objective; how well do the model's hyperparameters explain the observed data
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll) # runs the optimization; tunes the GP's internal hyperarameters to best explain train_x/train_y
    return model