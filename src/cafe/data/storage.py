"""Persist factual period rows, raw model arrays and scenario-level lineage."""

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from ..contracts import (ACTION_NAMES, FEATURE_NAMES, PRODUCTS, TARGET_NAMES, Action, ArtifactManifest,
                         DatasetArrays, DecisionInfo, Observation, PeriodRecord)
from .splits import SPLITS

OBS_COLUMNS = tuple(f"obs_{name}" for name in FEATURE_NAMES)
WASTE_COLUMNS = tuple(f"waste_{name}" for name in PRODUCTS)
COST_COLUMNS = ("revenue", "preparation_cost", "staff_cost", "profit")
COLUMNS = ("scenario_id", "trajectory_id", "period_index", "split", *OBS_COLUMNS, "action_id",
           *ACTION_NAMES, *TARGET_NAMES, "wait_valid", "entries", *WASTE_COLUMNS, *COST_COLUMNS, "diagnostics")


def _encode_diagnostic(value: object) -> dict | list:
    """Convert the simulator's optional decision record and prediction vector to JSON."""
    if isinstance(value, DecisionInfo):

        return asdict(value)

    if isinstance(value, np.ndarray):

        return value.tolist()

    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _load_diagnostics(encoded: str) -> dict:
    """Restore typed decision diagnostics and immutable end-of-period queue lengths."""
    diagnostics = json.loads(encoded)

    if not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be an object")

    decision = diagnostics.get("decision")

    if decision is not None:
        if decision["predicted_outcomes"] is not None:
            decision["predicted_outcomes"] = np.array(decision["predicted_outcomes"], dtype=float)

        diagnostics["decision"] = DecisionInfo(**decision)

    if "queue_lengths" in diagnostics:
        diagnostics["queue_lengths"] = tuple(diagnostics["queue_lengths"])

    return diagnostics


def save_records(records: Sequence[PeriodRecord], path: Path, *, splits: Mapping[str, str] | None = None) -> None:
    """Write the observation, applied action, realized outcome and accounting in one row."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)

        for row in records:
            action = row.action
            split = splits[row.scenario_id] if splits is not None else "unassigned"
            writer.writerow((row.scenario_id, row.trajectory_id, row.period_index, split,
                             *row.observation.values, action.action_id, *action.prices, action.staff,
                             *action.stock, *row.sales, row.abandonment, row.mean_wait, int(row.wait_valid),
                             row.entries, *row.waste, *(getattr(row, name) for name in COST_COLUMNS),
                             json.dumps(row.diagnostics, default=_encode_diagnostic, allow_nan=False)))


def load_records(path: Path) -> list[PeriodRecord]:
    """Reconstruct factual period records from their named CSV fields."""
    records = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"{path}: expected documented period columns")

        for row in reader:
            try:
                index = int(row["period_index"])
                obs = Observation(index, [float(row[name]) for name in OBS_COLUMNS])
                action = Action(int(row["action_id"]), tuple(float(row[k]) for k in ACTION_NAMES[:3]),
                                int(row["staff"]), tuple(int(row[k]) for k in ACTION_NAMES[4:]))
                sales = np.array([int(row[k]) for k in TARGET_NAMES[:3]])
                waste = np.array([int(row[k]) for k in WASTE_COLUMNS])
                wait_valid = int(row["wait_valid"])
                costs = [float(row[k]) for k in COST_COLUMNS]
                mean_wait = float(row["mean_wait"])

                if not row["scenario_id"] or not row["trajectory_id"] or None in row:
                    raise ValueError("expected scenario and trajectory IDs and the documented column count")

                if (wait_valid != int(sales.sum() > 0) or row["split"] not in (*SPLITS, "unassigned")
                        or not np.isfinite([*costs, mean_wait, *action.prices]).all()):
                    raise ValueError("invalid wait_valid, split or nonfinite numeric field")

                records.append(PeriodRecord(row["scenario_id"], row["trajectory_id"], index, obs, action,
                                            sales, int(row["entries"]), int(row["abandonment"]), mean_wait,
                                            bool(wait_valid), waste, *costs, _load_diagnostics(row["diagnostics"])))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}: invalid period row {reader.line_num}: {error}") from error

    return records


def save_arrays(arrays: DatasetArrays, path: Path) -> None:
    """Save raw numeric inputs, target masks and typed row identifiers."""
    np.savez_compressed(path, **{f.name: getattr(arrays, f.name) for f in fields(arrays)})


def load_arrays(path: Path, *, H: int, F: int) -> DatasetArrays:
    """Check the row keys, numeric shapes and factual target masks of one split."""
    with np.load(path, allow_pickle=False) as saved:
        required = {f.name for f in fields(DatasetArrays)}
        missing = required.difference(saved.files)

        if missing:
            raise ValueError(f"{path}: missing dataset fields {sorted(missing)}")

        arrays = DatasetArrays(**{name: saved[name] for name in required})

    keys = arrays.keys

    if (keys.ndim != 1 or keys.dtype.names != ("scenario_id", "trajectory_id", "period_index")
            or any(keys[name].dtype.kind != "U" for name in ("scenario_id", "trajectory_id"))
            or keys["period_index"].dtype.kind not in "iu" or (keys["period_index"] < 0).any()
            or (keys["scenario_id"] == "").any() or (keys["trajectory_id"] == "").any()
            or len(np.unique(keys)) != len(keys)):
        raise ValueError(f"{path}: expected unique scenario, trajectory and integer period keys")

    for name, shape in (("history", (H, F)), ("actions", (7,)), ("targets", (5,)), ("target_mask", (5,))):
        value = getattr(arrays, name)

        if value.shape != (len(keys), *shape) or value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise ValueError(f"{path}: {name} requires finite values of shape {(len(keys), *shape)}")

    mask = arrays.target_mask

    if mask.dtype != np.bool_ or not mask[:, :4].all() or not np.array_equal(
            mask[:, 4], arrays.targets[:, :3].sum(axis=1) > 0):
        raise ValueError(f"{path}: invalid factual target mask")

    return arrays


def save_manifest(manifest: ArtifactManifest, splits: Mapping[str, str], directory: Path) -> None:
    """Save scenario assignments alongside their artifact lineage."""
    for name, data in (("splits", dict(splits)), ("manifest", asdict(manifest))):
        (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_manifest(directory: Path, expected: ArtifactManifest) -> tuple[ArtifactManifest, dict[str, str]]:
    """Require matching provenance and a complete assignment of management scenarios."""
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((directory / "splits.json").read_text(encoding="utf-8"))

    if data != json.loads(json.dumps(asdict(expected))):
        raise ValueError(f"{directory}: incompatible dataset manifest")

    if set(splits) != set(expected.scenario_ids) or set(splits.values()) != set(SPLITS):
        raise ValueError(f"{directory}: invalid scenario splits")

    return expected, splits
