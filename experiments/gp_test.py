import torch
from bo.objective import sine, SINE_BOUNDS
from bo.gp import fit_gp

# a few training points across the domain
train_x = torch.tensor([[1.0], [3.0], [5.0], [8.0]], dtype=torch.double)
train_y = sine(train_x)

print("Training points:")
for xi, yi in zip(train_x, train_y):
    print(f"  x={xi.item():.2f}  y={yi.item():.4f}")

model = fit_gp(train_x, train_y, bounds=SINE_BOUNDS)

# Query the fitted model across the domain
test_x = torch.linspace(0, 10, 21, dtype=torch.double).unsqueeze(-1)
posterior = model.posterior(test_x)
mean = posterior.mean
variance = posterior.variance

print("\nPosterior predictions:")
print(f"{'x':>6} {'mean':>8} {'true f(x)':>10} {'variance':>10}")
true_y = sine(test_x)
for xi, mi, ti, vi in zip(test_x, mean, true_y, variance):
    print(f"{xi.item():6.2f} {mi.item():8.4f} {ti.item():10.4f} {vi.item():10.5f}")