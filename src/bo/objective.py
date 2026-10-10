"""Objective functions for the BO loop. 
Each function takes an (n, 1) tensor of inputs and returns an (n, 1) tensor of outputs. 
BO is considered as maximization, so functions where the target is a minimum (e.g. x^2) get negated.
"""
import torch

# --- sine --------------------------------------------------------------
SINE_BOUNDS = torch.tensor([[0.], [10.]], dtype=torch.double) 

def sine(x):
    return torch.sin(x)


# --- x squared --------------------------------------------------------------
XSQUARED_BOUNDS = torch.tensor([[-5.], [5.]], dtype=torch.double)

def x_squared(x):
    return -(x**2)

# --- levy --------------------------------------------------------------
from botorch.test_functions.synthetic import Levy as _BoTorchLevy
_levy = _BoTorchLevy(dim=1, negate=True)  

LEVY_BOUNDS = _levy.bounds.to(torch.double)
LEVY_OPTIMAL_VALUE = _levy.optimal_value


def levy(x):
    return _levy(x).unsqueeze(-1)