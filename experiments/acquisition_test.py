import torch
from bo.objective import sine, SINE_BOUNDS
from bo.gp import fit_gp
from bo.acquisition import propose_next_point

train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
train_y = sine(train_x)

model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

next_x = propose_next_point(model, train_y, SINE_BOUNDS)
print(f"Proposed next point: x = {next_x.item():.4f}")