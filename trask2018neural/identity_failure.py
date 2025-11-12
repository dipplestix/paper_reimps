"""Utilities for demonstrating numerical extrapolation failures in vanilla MLPs.

The implementation follows the setup described in *Trask et al. 2018* where an MLP is
trained to reproduce the identity function on a restricted interval.  The network fits
in-distribution values but extrapolates poorly when queried with inputs outside of the
training range.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn


def _make_mlp(depth: int, hidden_dim: int, activation: type[nn.Module]) -> nn.Sequential:
    """Construct a simple fully connected network with scalar input and output."""

    layers: list[nn.Module] = []
    in_features = 1
    for _ in range(depth):
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(activation())
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


@dataclass
class ExperimentConfig:
    """Configuration controlling the extrapolation failure experiment."""

    depth: int = 5
    hidden_dim: int = 5
    activation: type[nn.Module] = nn.ReLU
    train_range: Tuple[float, float] = (-5.0, 5.0)
    test_range: Tuple[float, float] = (-20.0, 20.0)
    n_train: int = 256
    n_eval: int = 512
    epochs: int = 4000
    lr: float = 1e-3
    device: torch.device = torch.device("cpu")
    weight_decay: float = 0.0
    batch_size: int = 64
    eval_points: Iterable[float] | None = None


class IdentityMLP(nn.Module):
    """A narrow MLP used to approximate the identity function."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.net = _make_mlp(config.depth, config.hidden_dim, config.activation)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - simple delegation
        return self.net(x)


@torch.no_grad()
def _make_dataset(n_samples: int, value_range: Tuple[float, float], device: torch.device) -> Tensor:
    low, high = value_range
    data = torch.empty(n_samples, 1, device=device).uniform_(low, high)
    return data


def run_experiment(config: ExperimentConfig, seed: int = 0) -> dict[str, Tensor]:
    """Train a narrow MLP on the identity task and evaluate inside/outside the range."""

    torch.manual_seed(seed)

    model = IdentityMLP(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    train_inputs = _make_dataset(config.n_train, config.train_range, config.device)
    train_targets = train_inputs.clone()

    history = torch.zeros(config.epochs)
    n_batches = max(1, config.n_train // config.batch_size)

    for epoch in range(config.epochs):
        perm = torch.randperm(config.n_train, device=config.device)
        epoch_loss = 0.0
        for batch_indices in perm.chunk(n_batches):
            batch_x = train_inputs[batch_indices]
            batch_y = train_targets[batch_indices]
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
        history[epoch] = epoch_loss / n_batches

    eval_inputs = config.eval_points
    if eval_inputs is None:
        eval_inputs = torch.linspace(*config.test_range, config.n_eval, device=config.device).unsqueeze(1)
    else:
        eval_inputs = torch.tensor(list(eval_inputs), device=config.device).float().unsqueeze(1)

    with torch.no_grad():
        train_preds = model(train_inputs)
        eval_preds = model(eval_inputs)

    return {
        "model": model,
        "train_inputs": train_inputs.cpu(),
        "train_targets": train_targets.cpu(),
        "train_predictions": train_preds.cpu(),
        "eval_inputs": eval_inputs.cpu(),
        "eval_predictions": eval_preds.cpu(),
        "loss_history": history.cpu(),
    }


def save_figure(outputs: dict[str, Tensor], out_path: Path) -> None:
    """Create a figure summarizing the extrapolation failure."""

    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(10, 4))

    ax_train.plot(outputs["loss_history"].numpy())
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("MSE Loss")
    ax_train.set_title("Training loss on [-5, 5]")

    eval_inputs = outputs["eval_inputs"].squeeze().numpy()
    eval_preds = outputs["eval_predictions"].squeeze().numpy()
    ax_eval.plot(eval_inputs, eval_inputs, label="Target", color="black")
    ax_eval.plot(eval_inputs, eval_preds, label="MLP prediction", color="tab:orange")
    ax_eval.scatter(outputs["train_inputs"].squeeze().numpy(), outputs["train_predictions"].squeeze().numpy(),
                    s=15, alpha=0.6, label="Train samples")
    ax_eval.set_xlabel("Input")
    ax_eval.set_ylabel("Output")
    ax_eval.set_title("Extrapolation outside training range")
    ax_eval.legend()

    fig.suptitle("Narrow MLP fails to extrapolate the identity function")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Entry-point used by the command line script."""

    config = ExperimentConfig()
    outputs = run_experiment(config)
    save_figure(outputs, Path("trask2018neural") / "outputs" / "identity_failure.png")


if __name__ == "__main__":  # pragma: no cover
    main()
