"""Fit numeric scales on real training periods before constructing windows."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..contracts import ACTION_NAMES, FEATURE_NAMES, TARGET_NAMES, PeriodRecord, ScalerStats
from .windows import numeric_action, targets


def fit_scalers(records: Sequence[PeriodRecord], splits: Mapping[str, str]) -> ScalerStats:
    """Count each training period once and use only valid waits for the wait scale."""
    rows = [row for row in records if splits[row.scenario_id] == "train"]
    keys = {(r.scenario_id, r.trajectory_id, r.period_index) for r in rows}
    waits = [row.mean_wait for row in rows if row.wait_valid]

    if len(keys) != len(rows) or not waits:
        raise ValueError("scalers require unique training periods and some valid training waits")

    observations = np.array([r.observation.values for r in rows])
    actions = np.array([numeric_action(r) for r in rows])
    outcomes = np.array([targets(r) for r in rows])
    mean = observations.mean(axis=0)
    std = np.maximum(1, observations.std(axis=0))
    flags = [FEATURE_NAMES.index(name) for name in ("wait_valid", "observation_valid")]
    mean[flags], std[flags] = 0, 1
    target_std = outcomes.std(axis=0)
    target_std[-1] = np.std(waits)

    return ScalerStats(mean, std, actions.mean(axis=0), np.maximum(1, actions.std(axis=0)),
                       np.maximum(1, target_std), tuple(sorted({r.scenario_id for r in rows})))


def save_scalers(scalers: ScalerStats, path: Path) -> None:
    """Save numeric transforms, their field order and training membership."""
    payload = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(scalers).items()}
    payload.update(feature_names=FEATURE_NAMES, action_names=ACTION_NAMES, target_names=TARGET_NAMES)
    path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")


def load_scalers(path: Path, expected_training_ids: Sequence[str]) -> ScalerStats:
    """Require finite field-aligned statistics fitted on the requested training IDs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = tuple(data["training_scenario_ids"])

    if ids != tuple(sorted(expected_training_ids)) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: incompatible training_scenario_ids")

    for name, fields in (("feature_names", FEATURE_NAMES), ("action_names", ACTION_NAMES),
                         ("target_names", TARGET_NAMES)):
        if data.get(name) != list(fields):
            raise ValueError(f"{path}: incompatible {name}")

    stats = {}

    for name, size in (("observation_mean", len(FEATURE_NAMES)), ("observation_std", len(FEATURE_NAMES)),
                       ("action_mean", len(ACTION_NAMES)), ("action_std", len(ACTION_NAMES)),
                       ("target_std", len(TARGET_NAMES))):
        values = np.asarray(data[name], dtype=float)

        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} requires {size} finite values")

        if name.endswith("_std") and (values < 1).any():
            raise ValueError(f"{path}: {name} below standard deviation floor")

        stats[name] = values

    return ScalerStats(**stats, training_scenario_ids=ids)
