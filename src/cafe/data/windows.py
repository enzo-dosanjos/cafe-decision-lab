"""Construct raw model inputs from actions and outcomes in the same trajectory."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ..contracts import DatasetArrays, PeriodRecord, make_history


def numeric_action(row: PeriodRecord) -> tuple[float, ...]:
    """Return prices, staff and stock in catalogue field order."""

    return (*row.action.prices, row.action.staff, *row.action.stock)


def targets(row: PeriodRecord) -> tuple[float, ...]:
    """Return factual sales, abandonment and served mean wait in target order."""

    return (*row.sales, row.abandonment, row.mean_wait)


def make_windows(records: Sequence[PeriodRecord], splits: Mapping[str, str], *,
                 split: str, H: int) -> DatasetArrays:
    """Include the current observation and at most H-1 previous rows of this day."""
    groups = defaultdict(list)

    for row in records:
        if splits[row.scenario_id] == split:
            groups[row.scenario_id, row.trajectory_id].append(row)

    if not groups:
        raise ValueError(f"No records for split {split!r}")

    # Within a configured experiment all trajectories cover the same complete day.
    if len({len(rows) for rows in groups.values()}) != 1:
        raise ValueError(f"{split}: unequal trajectory lengths indicate missing or duplicate periods")

    histories, actions, outcomes, masks, keys = [], [], [], [], []

    for (scenario, trajectory), rows in sorted(groups.items()):
        observations = []

        for index, row in enumerate(sorted(rows, key=lambda item: item.period_index)):
            if index != row.period_index or index != row.observation.period_index:
                raise ValueError(f"{scenario}/{trajectory}: expected consecutive period {index}")

            observations.append(row.observation)
            histories.append(make_history(observations, H))
            actions.append(numeric_action(row))
            outcomes.append(targets(row))
            masks.append((True, True, True, True, row.wait_valid))
            keys.append((scenario, trajectory, index))

    key_type = [("scenario_id", f"U{max(len(k[0]) for k in keys)}"),
                ("trajectory_id", f"U{max(len(k[1]) for k in keys)}"), ("period_index", "i8")]

    return DatasetArrays(np.asarray(histories), np.asarray(actions), np.asarray(outcomes),
                         np.asarray(masks), np.array(keys, dtype=key_type))
