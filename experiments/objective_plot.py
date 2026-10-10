import torch
import matplotlib.pyplot as plt
from bo.objective import sine, SINE_BOUNDS, x_squared, XSQUARED_BOUNDS, levy, LEVY_BOUNDS

# Fine grid across the domain
x_sine = torch.linspace(SINE_BOUNDS[0].item(), SINE_BOUNDS[1].item(), 200, dtype=torch.double).unsqueeze(-1)
y_sine = sine(x_sine)

x_sq = torch.linspace(XSQUARED_BOUNDS[0].item(), XSQUARED_BOUNDS[1].item(), 200, dtype=torch.double).unsqueeze(-1)
y_sq = x_squared(x_sq)

x_levy = torch.linspace(LEVY_BOUNDS[0].item(), LEVY_BOUNDS[1].item(), 200, dtype=torch.double).unsqueeze(-1)
y_levy = levy(x_levy)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x_sine.numpy(), y_sine.numpy())
axes[0].set_title("sine")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")

axes[1].plot(x_sq.numpy(), y_sq.numpy())
axes[1].set_title("x_squared (negated)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")

axes[2].plot(x_levy.numpy(), y_levy.numpy())
axes[2].set_title("x_levy (negated)")
axes[2].set_xlabel("x")
axes[2].set_ylabel("f(x)")

plt.tight_layout()
plt.savefig("objective_check.png")
print("Saved to objective_check.png")