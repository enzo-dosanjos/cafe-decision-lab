"""Run controlled validation-only experiments on the management dataset."""

import csv
from pathlib import Path
import argparse
import json
from cafe.cli import _load_dataset_policy, _catalogue_json
from cafe.contracts import build_catalogue, ScalerStats
from cafe.evaluate import select_managers
from cafe.managers.mlp import MLPManager
from cafe.simulation.scenarios import load_scenario

import numpy as np
import torch
from torch import nn
from cafe.cli import _model_inputs
from cafe.data.storage import load_arrays
from cafe.data.scaling import load_scalers
from cafe.models.base import OutcomePredictor
from cafe.models.lstm import LSTMPredictor
from cafe.models.mlp import MLPPredictor
from cafe.models.training import outcome_loss, _tensors


class ConvBlocks(nn.Module):
    """Compare shallow LeNet, stacked AlexNet/VGG, branching and skip connections in time."""

    def __init__(self, kind: str) -> None:
        """Keep four time steps and a common 32-channel representation."""
        super().__init__()
        self.kind = kind
        self.stem = nn.Conv1d(30, 32, 3, padding=1)
        depth = {"lenet": 1, "alexnet": 4, "vgg": 7, "resnet": 2, "googlenet": 1}[kind]
        activation = nn.Tanh if kind == "lenet" else nn.ReLU
        self.layers = nn.Sequential(
            *(
                op
                for _ in range(depth)
                for op in (nn.Conv1d(32, 32, 3, padding=1), activation())
            )
        )
        self.point = nn.Conv1d(32, 32, 1) if kind == "googlenet" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the temporal encoding after the architecture-specific block."""
        x = self.stem(x).relu()
        y = self.layers(x)

        if self.kind == "resnet":
            y = (x + y).relu()
        elif self.kind == "googlenet":
            y = torch.cat((y[:, :16], self.point(x)[:, 16:].relu()), dim=1)

        return nn.functional.adaptive_avg_pool1d(y, 2).flatten(1)


class TemporalPredictor(OutcomePredictor):
    """Apply CNN or vanilla recurrent layers along the observed time axis."""

    def __init__(self, stats: ScalerStats, kind: str) -> None:
        """Choose a temporal feature extractor for the common prediction head."""
        super().__init__(stats)
        self.kind = kind
        self.encoder = (
            nn.RNN(30, 32, batch_first=True)
            if kind == "rnn"
            else nn.Sequential(
                nn.Conv1d(30, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten(),
            )
        )
        if kind in ("lenet", "alexnet", "vgg", "resnet", "googlenet"):
            self.encoder = ConvBlocks(kind)

        self.head = nn.Sequential(
            nn.Linear((32 if kind == "rnn" else 64) + 7, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward_logits(
        self, history: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Combine a causal window encoding with the current candidate action."""
        if self.kind == "rnn":
            encoded = self.encoder(history)[0][:, -1]
        else:
            encoded = self.encoder(history.transpose(1, 2))

        return self.head(torch.cat((encoded, actions), dim=1))


SENSITIVITY = (
    "narrow",
    "wide",
    "sigmoid",
    "tanh",
    "leaky",
    "prelu",
    "softplus",
    "silu",
    "gelu",
    "xavier",
    "he",
    "batch32",
    "batch512",
    "dropout-small",
    "dropout-large",
    "l2-small",
    "l2-large",
)


def make_model(name: str, stats: ScalerStats, *, history_length: int = 4) -> OutcomePredictor:
    """Construct one architecture with a shared raw-unit prediction contract."""
    if history_length != 4 and (name not in ("rnn", "adam") or history_length != 1):
        raise ValueError("the one-observation ablation supports the vanilla RNN and baseline MLP")

    model = MLPPredictor(stats, H=history_length)

    if name == "affine":
        model.network = nn.Linear(127, 5)
    elif name in ("rnn", "cnn", "lenet", "alexnet", "vgg", "resnet", "googlenet"):
        model = TemporalPredictor(stats, name)
    elif name == "lstm":
        model = LSTMPredictor(stats)
    elif name == "shallow":
        model.network = nn.Sequential(
            nn.Linear(127, 32), nn.Sigmoid(), nn.Linear(32, 5)
        )
    elif name in ("narrow", "wide"):
        model = MLPPredictor(
            stats, hidden_sizes=(32, 16) if name == "narrow" else (128, 64)
        )
    elif name in ("xavier", "he"):
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                if name == "he":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(layer.weight)
    elif name in ("sigmoid", "tanh", "leaky", "prelu", "softplus", "silu", "gelu"):
        activation = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "softplus": nn.Softplus,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }[name]
        model.network[1], model.network[3] = activation(), activation()
    elif name in ("dropout", "dropout-small", "dropout-large"):
        model.network.insert(
            2,
            nn.Dropout(
                {"dropout": 0.2, "dropout-small": 0.1, "dropout-large": 0.4}[name]
            ),
        )
    elif name == "batchnorm":
        model.network.insert(1, nn.BatchNorm1d(64))

    return model


def run(
    name: str,
    rate: float,
    seed: int,
    training: tuple[torch.Tensor, ...],
    validation: tuple[torch.Tensor, ...],
    stats: ScalerStats,
    epochs: int,
    output: Path,
    *, history_length: int = 4,
) -> tuple[float, int]:
    """Fit one candidate and retain the earliest minimum validation loss."""
    torch.manual_seed(seed)
    model = make_model(name, stats, history_length=history_length)
    n = len(training[0])
    batch = {"gd": n, "single-sgd": 1, "batch32": 32, "batch512": 512}.get(name, 128)
    optimizer = {
        "gd": torch.optim.SGD,
        "single-sgd": torch.optim.SGD,
        "mini-sgd": torch.optim.SGD,
        "momentum": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
    }.get(name, torch.optim.Adam)
    options = (
        {"momentum": 0.9}
        if name == "momentum"
        else {"weight_decay": 0.001}
        if name == "l2"
        else {}
    )
    if name in ("l2-small", "l2-large"):
        options = {"weight_decay": 0.0001 if name == "l2-small" else 0.01}

    opt = optimizer(model.parameters(), lr=rate, **options)
    schedules = torch.optim.lr_scheduler
    scheduler = {
        "step": lambda: schedules.StepLR(opt, max(1, epochs // 2), gamma=0.3),
        "exp": lambda: schedules.ExponentialLR(opt, 0.9),
        "cosine": lambda: schedules.CosineAnnealingLR(opt, epochs),
        "restart": lambda: schedules.CosineAnnealingWarmRestarts(
            opt, max(1, epochs // 2)
        ),
    }.get(name)
    scheduler = scheduler() if scheduler else None
    best = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for idx in torch.randperm(n).split(batch):
            h, a, y, mask = (v[idx] for v in training)
            opt.zero_grad()
            loss = outcome_loss(model(h, a), y, mask, model.target_std)
            if name == "l1":
                loss = loss + 0.0001 * sum(
                    p.abs().sum() for p in model.parameters() if p.ndim > 1
                )

            loss.backward()

            if name in ("rnn", "lstm"):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

        if scheduler:
            scheduler.step()

        h, a, y, mask = validation
        score = outcome_loss(model.predict(h, a), y, mask, model.target_std).item()

        if score < best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), output / f"{name}-{rate}-{seed}-{epochs}.pt")

    return best, best_epoch


def main() -> None:
    """Run the forecast screen, then compare shortlisted managers on matched validation days."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--customer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    config, manifest, splits = _model_inputs(root)
    train_ids = sorted(s for s in splits if splits[s] == "train")
    stats = load_scalers(root / "dataset/scalers.json", train_ids)
    train = load_arrays(root / "dataset/train.npz", H=4, F=30)
    validation = _tensors(
        load_arrays(root / "dataset/validation.npz", H=4, F=30), "cpu"
    )
    full = _tensors(train, "cpu")
    subset = np.random.default_rng(518).choice(train_ids, 64, replace=False)
    pilot = tuple(v[np.isin(train.keys["scenario_id"], subset)] for v in full)
    names = "adam affine shallow cnn rnn lstm gd single-sgd mini-sgd momentum adagrad rmsprop l1 l2 dropout batchnorm step exp cosine restart".split()
    torch.set_num_threads(1)
    results = []

    with (output / "validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("stage", "method", "learning_rate", "seed", "loss", "best_epoch")
        )

        for name in names:
            rates = (
                (0.001, 0.01)
                if name
                in (
                    "gd",
                    "single-sgd",
                    "mini-sgd",
                    "momentum",
                    "adagrad",
                    "rmsprop",
                    "adam",
                )
                else (0.001,)
            )

            for rate in rates:
                loss, epoch = run(name, rate, 0, pilot, validation, stats, 8, output)
                writer.writerow(("pilot", name, rate, 0, loss, epoch))
                stream.flush()
                results.append((loss, name, rate))

        winners = {min(results)[1:]}
        for group in (
            {"adam", "shallow", "affine", "cnn", "rnn", "lstm"},
            {"l1", "l2", "dropout", "batchnorm"},
            {"step", "exp", "cosine", "restart"},
            {"gd", "single-sgd", "mini-sgd", "momentum", "adagrad", "rmsprop"},
        ):
            winners.add(min(r for r in results if r[1] in group)[1:])
        winners.add(("adam", 0.001))
        winners.update(
            (name, 0.001) for name in ("affine", "shallow", "cnn", "rnn", "lstm")
        )

        for name, rate in sorted(winners):
            for seed in range(3):
                loss, epoch = run(name, rate, seed, full, validation, stats, 35, output)
                writer.writerow(("full", name, rate, seed, loss, epoch))
                stream.flush()

    with (output / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "loss", "epoch"))

        for name in SENSITIVITY:
            writer.writerow(
                (name, *run(name, 0.001, 0, pilot, validation, stats, 8, output))
            )

    # Historical vision networks are represented by reduced temporal blocks.
    with (output / "cnn-examples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("temporal_adaptation", "loss", "epoch"))

        for name in ("lenet", "alexnet", "vgg", "resnet", "googlenet"):
            writer.writerow(
                (name, *run(name, 0.001, 0, pilot, validation, stats, 8, output))
            )

    catalogue = build_catalogue(config)
    policy, _, _ = _load_dataset_policy(
        config, args.customer_dir, _catalogue_json(catalogue)
    )

    if policy.fingerprint() != manifest.parent_hashes["customer_policy"]:
        raise ValueError("customer policy must match the management dataset")

    days = [
        load_scenario(root / "scenarios" / f"{s}.npz")
        for s in sorted(splits)
        if splits[s] == "validation"
    ]

    for name, rate in sorted(winners):
        for seed in range(3):
            model = make_model(name, stats)
            model.load_state_dict(
                torch.load(output / f"{name}-{rate}-{seed}-35.pt", weights_only=True)
            )
            manager = MLPManager(model, catalogue, config)
            _, trials = select_managers({"model": [manager]}, days, policy, config)
            (output / f"business-{name}-{rate}-{seed}.json").write_text(
                json.dumps(trials[0])
            )


if __name__ == "__main__":
    main()
