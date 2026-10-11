import torch
from bo.objective import sine, SINE_BOUNDS
from bo.gp import fit_gp


def test_fit_gp_runs_without_error():
    train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
    train_y = sine(train_x)

    model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

    test_x = torch.tensor([[2.0], [6.0]], dtype=torch.double)
    posterior = model.posterior(test_x)
    assert posterior.mean.shape == test_x.shape
    assert posterior.variance.shape == test_x.shape


def test_gp_predicts_near_training_points():
    train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
    train_y = sine(train_x)

    model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

    # query at the exact training x-values; the mean should be close to the true observed y there.
    posterior = model.posterior(train_x)
    assert torch.allclose(posterior.mean, train_y, atol=0.05)


def test_gp_more_confident_near_data_than_far_from_it():
    train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
    train_y = sine(train_x)

    model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

    near_point = torch.tensor([[5.0]], dtype=torch.double)   # right on a training point
    far_point = torch.tensor([[10.0]], dtype=torch.double)   # past the last training point

    var_near = model.posterior(near_point).variance
    var_far = model.posterior(far_point).variance

    assert var_near.item() < var_far.item()