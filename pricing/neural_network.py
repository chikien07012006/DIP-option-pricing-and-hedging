from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "configs" / "pricing.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FEATURES = [
    "spot_price",
    "strike_price",
    "barrier_to_spot_ratio",
    "time_to_maturity",
    "volatility",
    "risk_free_rate",
]
TARGET = "analytical_price"

RELATIVE_ERROR_MIN_PRICE = 0.1

class Standardizer:

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=0), x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Standardizer":
        return cls(np.asarray(state["mean"]), np.asarray(state["std"]))


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame[TARGET].to_numpy(dtype=np.float32)
    return x, y, frame


class DIPPricingMLP(nn.Module):

    def __init__(self, input_dim: int = 6, hidden_layers: tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for width in hidden_layers:
            layers += [nn.Linear(in_features, width), nn.ReLU()]
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def evaluate_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 8192) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = model(x[start:start + batch_size])
            total += torch.sum((pred - y[start:start + batch_size]) ** 2).item()
    return total / len(x)


def train(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        running = 0.0
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)

        train_loss = running / len(x_train)
        val_loss = evaluate_mse(model, x_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:4d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return history


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    mask = y_true > RELATIVE_ERROR_MIN_PRICE
    relative = np.abs(error[mask]) / y_true[mask]
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        f"mean_relative_error(price>{RELATIVE_ERROR_MIN_PRICE})": float(np.mean(relative)) if mask.any() else float("nan"),
        "negative_predictions": int(np.sum(y_pred < 0)),
    }


def error_breakdown(frame: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    """MAE / RMSE by moneyness, volatility, maturity, and barrier proximity."""
    frame = frame.assign(
        moneyness_K_over_S=frame["strike_price"] / frame["spot_price"],
        abs_error=(frame["predicted_price"] - frame[TARGET]).abs(),
    )
    groups = {
        "moneyness_K_over_S": "moneyness_K_over_S",
        "volatility": "volatility",
        "time_to_maturity": "time_to_maturity",
        "barrier_to_spot_ratio": "barrier_to_spot_ratio",
    }
    rows = []
    for label, column in groups.items():
        bins = pd.qcut(frame[column], q=num_bins)
        for interval, group in frame.groupby(bins, observed=True):
            rows.append(
                {
                    "feature": label,
                    "bin": f"({interval.left:.3f}, {interval.right:.3f}]",
                    "n": len(group),
                    "mean_price": group[TARGET].mean(),
                    "mae": group["abs_error"].mean(),
                    "rmse": np.sqrt((group["abs_error"] ** 2).mean()),
                }
            )
    return pd.DataFrame(rows)


def predict(model: nn.Module, x: torch.Tensor) -> tuple[np.ndarray, float]:
    """Return predictions and inference time in microseconds per contract."""
    model.eval()
    with torch.no_grad():
        model(x[:1024])  # warm-up
        start = time.perf_counter()
        pred = model(x)
        elapsed = time.perf_counter() - start
    return pred.cpu().numpy(), elapsed / len(x) * 1e6


def plot_learning_curves(history: dict, output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train MSE")
    plt.plot(epochs, history["val_loss"], label="Validation MSE")
    plt.axvline(history["best_epoch"], color="grey", linestyle="--", linewidth=1,
                label=f"Best epoch ({history['best_epoch']})")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (price units²)")
    plt.title("MLP pricing surrogate: learning curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    limit = max(y_true.max(), y_pred.max())
    axes[0].scatter(y_true, y_pred, s=2, alpha=0.3)
    axes[0].plot([0, limit], [0, limit], color="red", linewidth=1)
    axes[0].set_xlabel("Analytical price")
    axes[0].set_ylabel("MLP price")
    axes[0].set_title("Test set: predicted vs analytical")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(y_pred - y_true, bins=100)
    axes[1].set_xlabel("Prediction error (MLP - analytical)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Test set: error distribution")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def load_nn_config() -> dict:
    with open(CONFIG_PATH) as handle:
        return yaml.safe_load(handle)["neural_network"]


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[DIPPricingMLP, Standardizer]:
    checkpoint = torch.load(path, map_location=device)
    model = DIPPricingMLP(len(checkpoint["features"]), tuple(checkpoint["hidden_layers"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, Standardizer.from_state_dict(checkpoint["scaler"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the DIP pricing MLP.")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda (default: cpu).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_nn_config()
    arch = config["architecture"]
    training = config["training"]
    epochs = args.epochs or training["epochs"]
    seed = training["random_seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)

    x_train, y_train, _ = load_split("train")
    x_val, y_val, _ = load_split("validation")
    x_test, y_test, test_frame = load_split("test")

    scaler = Standardizer.fit(x_train)

    def to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)

    xt, yt = to_tensor(scaler.transform(x_train)), to_tensor(y_train)
    xv, yv = to_tensor(scaler.transform(x_val)), to_tensor(y_val)
    xs = to_tensor(scaler.transform(x_test))

    model = DIPPricingMLP(len(FEATURES), tuple(arch["hidden_layers"])).to(device)
    print(model)
    print(f"Train {len(xt):,} | validation {len(xv):,} | test {len(xs):,} | device {device}")

    start = time.perf_counter()
    history = train(
        model, xt, yt, xv, yv,
        learning_rate=training["learning_rate"],
        batch_size=training["batch_size"],
        epochs=epochs,
        patience=training["patience"],
        seed=seed,
    )
    training_seconds = time.perf_counter() - start

    # Evaluation on the raw price scale.
    train_pred, _ = predict(model, xt)
    val_pred, _ = predict(model, xv)
    test_pred, us_per_contract = predict(model, xs)

    metrics = {
        "train": regression_metrics(y_train, train_pred),
        "validation": regression_metrics(y_val, val_pred),
        "test": regression_metrics(y_test, test_pred),
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["train_loss"]),
        "training_seconds": training_seconds,
        "inference_microseconds_per_contract": us_per_contract,
    }

    test_frame = test_frame.assign(predicted_price=test_pred)
    breakdown = error_breakdown(test_frame)

    # Save artefacts.
    for directory in (CHECKPOINT_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "features": FEATURES,
            "hidden_layers": list(arch["hidden_layers"]),
            "scaler": scaler.state_dict(),
            "history": history,
        },
        CHECKPOINT_DIR / "dip_mlp.pt",
    )
    with open(TABLE_DIR / "nn_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    breakdown.to_csv(TABLE_DIR / "nn_error_breakdown.csv", index=False, float_format="%.6f")
    test_frame.to_csv(TABLE_DIR / "nn_test_predictions.csv", index=False, float_format="%.8f")
    plot_learning_curves(history, FIGURE_DIR / "nn_learning_curves.png")
    plot_predictions(y_test, test_pred, FIGURE_DIR / "nn_test_predictions.png")

    print("\n=== Metrics (raw price scale) ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== Test error breakdown ===")
    print(breakdown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArtefacts saved under {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()
