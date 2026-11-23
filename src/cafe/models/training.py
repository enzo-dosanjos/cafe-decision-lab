"""Fit supervised outcome forecasts and retain the lowest validation-loss checkpoint."""

from collections.abc import Sequence
import csv
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from ..config import Config, config_hash
from ..contracts import ACTION_NAMES, TARGET_NAMES, ArtifactManifest, DatasetArrays, ScalerStats
from .base import OutcomePredictor
from .mlp import MLPPredictor
from .lstm import LSTMPredictor


def outcome_loss(predictions: Tensor, targets: Tensor, target_mask: Tensor, target_std: Tensor) -> Tensor:
    """Average normalized sales, abandonment and valid-wait squared errors by group."""
    errors = ((predictions - targets) / target_std).square()
    valid = target_mask[:, 4]

    return errors[:, :3].mean() + errors[:, 3].mean() + (errors[:, 4] * valid).sum() / valid.sum().clamp_min(1)


def _tensors(arrays: DatasetArrays, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Move one raw split to the training device, preserving the Boolean mask."""

    return (torch.as_tensor(arrays.history, dtype=torch.float32, device=device),
            torch.as_tensor(arrays.actions, dtype=torch.float32, device=device),
            torch.as_tensor(arrays.targets, dtype=torch.float32, device=device),
            torch.as_tensor(arrays.target_mask, dtype=torch.bool, device=device))


def _metrics(model: OutcomePredictor, data: Sequence[Tensor]) -> dict[str, float]:
    """Measure the full split with the correct denominator for each original-unit MAE."""
    history, actions, targets, mask = data
    predicted = model.predict(history, actions)
    errors = (predicted - targets).abs()
    metrics = {"loss": float(outcome_loss(predicted, targets, mask, model.target_std))}

    for i, name in enumerate(TARGET_NAMES):
        values = errors[:, i][mask[:, i]]
        metrics[f"mae_{name}"] = float(values.mean()) if len(values) else float("nan")

    return metrics


def save_checkpoint(model: OutcomePredictor, path: Path, config: Config, seed: int, manifest: ArtifactManifest) -> None:
    """Save reconstructible architecture, model buffers, settings and dataset lineage."""
    if isinstance(model, MLPPredictor):
        architecture = {"architecture": "mlp", "H": model.H, "F": model.F, "hidden_sizes": model.hidden_sizes}
    elif isinstance(model, LSTMPredictor):
        architecture = {"architecture": "lstm", "F": model.F, "hidden_size": model.hidden_size,
                        "head_size": model.head_size}
    else:
        raise ValueError("unsupported checkpoint architecture")

    torch.save({**architecture,
                "state": model.state_dict(), "config": asdict(config), "seed": seed, "manifest": asdict(manifest),
                "action_names": ACTION_NAMES, "training_scenario_ids": model.training_scenario_ids,
                "torch_version": str(torch.__version__), "device": str(model.action_mean.device)}, path)


def load_checkpoint(path: Path, expected: ArtifactManifest, *, device: str = "cpu") -> OutcomePredictor:
    """Restore a predictor only when its parent artifacts and numeric action order match."""
    saved = torch.load(path, map_location="cpu", weights_only=True)

    if (saved["architecture"] not in ("mlp", "lstm") or tuple(saved["action_names"]) != ACTION_NAMES
            or config_hash(Config(**saved["config"])) != expected.config_hash
            or json.dumps(saved["manifest"], sort_keys=True) != json.dumps(asdict(expected), sort_keys=True)):
        raise ValueError(f"{path}: incompatible model architecture or dataset lineage")

    state = saved["state"]
    stats = ScalerStats(*(state[k].numpy() for k in ("observation_mean", "observation_std", "action_mean",
                                                      "action_std", "target_std")), tuple(saved["training_scenario_ids"]))

    if saved["architecture"] == "mlp":
        model = MLPPredictor(stats, H=saved["H"], F=saved["F"], hidden_sizes=tuple(saved["hidden_sizes"]))
    else:
        model = LSTMPredictor(stats, F=saved["F"], hidden_size=saved["hidden_size"], head_size=saved["head_size"])

    model.load_state_dict(state)

    return model.to(device).eval()


def train_predictor(model_name: str, train: DatasetArrays, validation: DatasetArrays, scalers: ScalerStats,
                    config: Config, *, seed: int, device: str, output_dir: Path,
                    manifest: ArtifactManifest) -> OutcomePredictor:
    """Train with seeded mini-batches and Adam, restoring the earliest best validation epoch."""
    if model_name not in ("mlp", "lstm"):
        raise ValueError(f"unsupported model: {model_name}")

    train_ids = set(train.keys["scenario_id"])
    validation_ids = set(validation.keys["scenario_id"])

    if (train_ids != set(scalers.training_scenario_ids) or train_ids & validation_ids or not validation_ids
            or not (train_ids | validation_ids).issubset(manifest.scenario_ids) or config_hash(config) != manifest.config_hash):
        raise ValueError("expected disjoint train/validation scenarios and matching training scalers")

    torch.manual_seed(seed)

    if model_name == "mlp":
        model = MLPPredictor(scalers, H=config.model["history_length"], F=config.model["feature_count"])
    else:
        model = LSTMPredictor(scalers, F=config.model["feature_count"])

    model = model.to(device)
    settings = config.training
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    training = _tensors(train, device)
    valid = _tensors(validation, device)
    best_loss = float("inf")
    best_epoch = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"seed-{seed}.pt"
    start = perf_counter()
    epoch_log = output_dir / f"seed-{seed}-epochs.csv"

    if checkpoint.exists() or epoch_log.exists():
        raise FileExistsError(checkpoint)

    # Publish only a completed run. Hard links also prevent concurrent writers replacing a seed.
    with TemporaryDirectory(dir=output_dir) as tmp:
        best_file = Path(tmp) / checkpoint.name
        temporary_log = Path(tmp) / epoch_log.name

        with temporary_log.open("w", newline="") as stream:
            writer = None

            for epoch in range(1, settings["max_epochs"] + 1):
                model.train()
                order = torch.randperm(len(train.history), device=device)

                for indices in order.split(settings["batch_size"]):
                    history, actions, targets, mask = (t[indices] for t in training)
                    optimizer.zero_grad()
                    loss = outcome_loss(model(history, actions), targets, mask, model.target_std)
                    loss.backward()

                    if model_name == "lstm":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])

                    optimizer.step()

                row = {"epoch": epoch}

                for label, tensors in (("train", training), ("validation", valid)):
                    row.update({f"{label}_{key}": value for key, value in _metrics(model, tensors).items()})

                score = row["validation_loss"]

                if not np.isfinite(score):
                    raise ValueError(f"nonfinite validation loss at epoch {epoch}")

                if score < best_loss:
                    best_loss, best_epoch = score, epoch
                    save_checkpoint(model, best_file, config, seed, manifest)

                row.update(best_epoch=best_epoch, elapsed_seconds=perf_counter() - start)

                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()

                writer.writerow(row)
                stream.flush()

                if epoch - best_epoch >= settings["patience"]:
                    break

        epoch_log.hardlink_to(temporary_log)

        try:
            checkpoint.hardlink_to(best_file)
        except OSError:
            epoch_log.unlink()
            raise

    return load_checkpoint(checkpoint, manifest, device=device)
