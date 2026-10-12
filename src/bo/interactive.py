"""Interactive visualization of a full BO run"""
import torch
import plotly.graph_objects as go

from .gp import fit_gp
from .acquisition import propose_next_point


def build_interactive_plot(objective, bounds, n_init=4, n_iters=10, seed=7,
                            resolution=200, outpath="outputs/interactive.html"):
    torch.manual_seed(seed)

    dim = bounds.shape[-1]
    assert dim == 1, "interactive plot only supports 1D objectives"
    lower, upper = bounds[0], bounds[1]

    train_x = lower + (upper - lower) * torch.rand(n_init, dim, dtype=bounds.dtype)
    train_y = objective(train_x)

    test_x = torch.linspace(lower.item(), upper.item(), resolution, dtype=torch.double).unsqueeze(-1)
    test_x_np = test_x.squeeze(-1).numpy()

    frames_data = []

    def record_frame(model, train_x, train_y, new_x=None, new_y=None, label=""):
        posterior = model.posterior(test_x)
        mean = posterior.mean.detach().squeeze(-1).numpy()
        std = posterior.variance.detach().sqrt().squeeze(-1).numpy()
        frames_data.append(dict(
            mean=mean, std=std,
            train_x=train_x.squeeze(-1).numpy().copy(),
            train_y=train_y.squeeze(-1).numpy().copy(),
            new_x=new_x, new_y=new_y,
            label=label,
        ))

    model = fit_gp(train_x, train_y, bounds=bounds)
    record_frame(model, train_x, train_y, label="Initial random points")

    for it in range(1, n_iters + 1):
        next_x = propose_next_point(model, train_y, bounds)
        next_y = objective(next_x)

        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])
        model = fit_gp(train_x, train_y, bounds=bounds)

        record_frame(model, train_x, train_y,
                      new_x=next_x.item(), new_y=next_y.item(),
                      label=f"Iteration {it}")

    _write_html(test_x_np, frames_data, outpath)
    return frames_data


def _write_html(test_x_np, frames_data, outpath):
    frames = []
    for i, d in enumerate(frames_data):
        upper_band = d["mean"] + 2 * d["std"]
        lower_band = d["mean"] - 2 * d["std"]

        data = [
            go.Scatter(x=test_x_np, y=upper_band, mode="lines",
                       line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=test_x_np, y=lower_band, mode="lines",
                       line=dict(width=0), fill="tonexty",
                       fillcolor="rgba(99,110,250,0.2)",
                       name="±2 std", hoverinfo="skip"),
            go.Scatter(x=test_x_np, y=d["mean"], mode="lines",
                       line=dict(color="rgb(99,110,250)", width=2),
                       name="GP mean"),
            go.Scatter(x=d["train_x"], y=d["train_y"], mode="markers",
                       marker=dict(color="black", size=8),
                       name="observed points"),
        ]
        new_x_vals = [d["new_x"]] if d["new_x"] is not None else []
        new_y_vals = [d["new_y"]] if d["new_y"] is not None else []
        data.append(go.Scatter(
            x=new_x_vals, y=new_y_vals, mode="markers",
            marker=dict(color="red", size=14, symbol="star"),
            name="just evaluated",
        ))
        frames.append(go.Frame(data=data, name=str(i), layout=go.Layout(title=d["label"])))

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=frames_data[0]["label"],
            xaxis_title="x", yaxis_title="f(x)",
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[str(i)], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=True))],
                            label=str(i))
                       for i in range(len(frames))],
                active=0,
                currentvalue=dict(prefix="Iteration: "),
            )],
        ),
        frames=frames,
    )
    fig.write_html(outpath)
    print(f"Saved interactive plot to {outpath}")