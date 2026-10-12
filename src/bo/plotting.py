"""Diagnostic plotting for the GP surrogate: mean, ~95% uncertainty band, and the training points observed so far."""
import os
import torch
import matplotlib.pyplot as plt


def plot_gp(model, train_x, train_y, bounds, iteration, outdir="outputs"):
    """Plot the GP's posterior mean +/- 2 std across the domain, with
    current training points marked.
    """
    os.makedirs(outdir, exist_ok=True)

    lower, upper = bounds[0].item(), bounds[1].item()
    test_x = torch.linspace(lower, upper, 200, dtype=torch.double).unsqueeze(-1)

    posterior = model.posterior(test_x)
    mean = posterior.mean.detach().squeeze(-1)
    std = posterior.variance.detach().sqrt().squeeze(-1)

    x_np = test_x.squeeze(-1).numpy()
    mean_np = mean.numpy()
    upper_band = (mean + 2 * std).numpy()
    lower_band = (mean - 2 * std).numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(x_np, mean_np, label="GP mean")
    plt.fill_between(x_np, lower_band, upper_band, alpha=0.3, label="±2 std (~95%)")
    plt.scatter(train_x.squeeze(-1).numpy(), train_y.squeeze(-1).numpy(),
                color="black", zorder=5, label="observed points")
    plt.title(f"GP state — iteration {iteration}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"gp_iter_{iteration:03d}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

def plot_convergence(best_so_far, true_optimum=None, function_name="", outpath="outputs/convergence.png"):
    """Plot best-value-found vs. iteration"""
    iterations = list(range(len(best_so_far)))

    plt.figure(figsize=(8, 4))
    plt.plot(iterations, best_so_far, marker="o", label="best found so far")

    if true_optimum is not None:
        plt.axhline(true_optimum, color="red", linestyle="--", label="true optimum")

    plt.xlabel("Iteration")
    plt.ylabel("Best value found")
    title = f"Convergence — {function_name}" if function_name else "Convergence"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")