import torch
from bo.objective import sine, SINE_BOUNDS
from bo.gp import fit_gp
from bo.acquisition import propose_next_point


def test_propose_next_point_shape():
    train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
    train_y = sine(train_x)
    model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

    next_x = propose_next_point(model, train_y, SINE_BOUNDS)

    assert next_x.shape == (1, 1)


def test_propose_next_point_within_bounds():
    train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
    train_y = sine(train_x)
    model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

    next_x = propose_next_point(model, train_y, SINE_BOUNDS)

    lower, upper = SINE_BOUNDS[0].item(), SINE_BOUNDS[1].item()
    assert lower <= next_x.item() <= upper