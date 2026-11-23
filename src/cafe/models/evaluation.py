"""Measure raw-unit forecast errors on factual exploratory trajectories."""

import numpy as np
import torch

from ..contracts import DatasetArrays, TARGET_NAMES
from .base import OutcomePredictor


def evaluate_forecasts(model: OutcomePredictor, arrays: DatasetArrays, *,
                       batch_size: int = 128) -> tuple[list[dict], dict[str, float | None]]:
    """Return keyed predictions and masked per-target MAE/RMSE in original units."""
    if batch_size < 1 or not len(arrays.history):
        raise ValueError("forecast evaluation requires data and a positive batch_size")

    device = model.action_mean.device
    predictions = []

    for start in range(0, len(arrays.history), batch_size):
        history = torch.as_tensor(arrays.history[start:start + batch_size], dtype=torch.float32, device=device)
        actions = torch.as_tensor(arrays.actions[start:start + batch_size], dtype=torch.float32, device=device)
        predictions.append(model.predict(history, actions).cpu().numpy())

    predicted = np.concatenate(predictions)
    rows = []
    metrics = {}

    for index, key in enumerate(arrays.keys):
        row = {"scenario_id": str(key["scenario_id"]), "trajectory_id": str(key["trajectory_id"]),
               "period_index": int(key["period_index"])}

        for j, name in enumerate(TARGET_NAMES):
            row[f"predicted_{name}"] = float(predicted[index, j])
            row[f"observed_{name}"] = float(arrays.targets[index, j]) if arrays.target_mask[index, j] else None

        rows.append(row)

    for j, name in enumerate(TARGET_NAMES):
        errors = (predicted[:, j] - arrays.targets[:, j])[arrays.target_mask[:, j]]
        metrics[f"mae_{name}"] = float(np.abs(errors).mean()) if len(errors) else None
        metrics[f"rmse_{name}"] = float(np.sqrt(np.square(errors).mean())) if len(errors) else None

    return rows, metrics
