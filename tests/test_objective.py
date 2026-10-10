import torch
from bo.objective import (
    sine, SINE_BOUNDS,
    x_squared, XSQUARED_BOUNDS,
    levy, LEVY_BOUNDS, LEVY_OPTIMAL_VALUE,
)


def test_sine_shape():
    x = torch.tensor([[0.], [1.571], [5.]], dtype=torch.double)
    y = sine(x)
    assert y.shape == x.shape


def test_sine_known_values():
    x = torch.tensor([[0.], [1.5707963267948966], [7.853981633974483]], dtype=torch.double)
    y = sine(x)
    expected = torch.tensor([[0.], [1.], [1.]], dtype=torch.double)
    assert torch.allclose(y, expected, atol=1e-6)


def test_x_squared_shape():
    x = torch.tensor([[-5.], [0.], [5.]], dtype=torch.double)
    y = x_squared(x)
    assert y.shape == x.shape


def test_x_squared_known_values():
    x = torch.tensor([[-5.], [-2.], [0.], [2.], [5.]], dtype=torch.double)
    y = x_squared(x)
    expected = torch.tensor([[-25.], [-4.], [0.], [-4.], [-25.]], dtype=torch.double)
    assert torch.allclose(y, expected)


def test_x_squared_is_symmetric():
    x_pos = torch.tensor([[1.], [2.], [5.]], dtype=torch.double)
    assert torch.allclose(x_squared(x_pos), x_squared(-x_pos))


def test_levy_shape():
    x = torch.tensor([[-5.], [0.], [5.]], dtype=torch.double)
    y = levy(x)
    assert y.shape == x.shape


def test_levy_known_optimum():
    # Levy's documented global optimum is at x=1, for every dimension.
    x = torch.tensor([[1.0]], dtype=torch.double)
    y = levy(x)
    expected = torch.tensor([[LEVY_OPTIMAL_VALUE]], dtype=torch.double)
    assert torch.allclose(y, expected, atol=1e-6)


def test_bounds_shapes():
    assert SINE_BOUNDS.shape == (2, 1)
    assert XSQUARED_BOUNDS.shape == (2, 1)
    assert LEVY_BOUNDS.shape == (2, 1)