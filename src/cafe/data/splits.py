"""Keep all trajectories of a management scenario in one seeded split."""

from collections.abc import Mapping, Sequence

import numpy as np

from ..contracts import PeriodRecord

SPLITS = ("train", "validation", "test")


def make_split_manifest(scenario_ids: Sequence[str], *, seed: int,
                        fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, str]:
    """Shuffle sorted IDs, allocating leftover scenarios by largest fractional share."""
    ids = sorted(scenario_ids)
    shares = np.asarray(fractions, dtype=float)

    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("scenario_ids must be unique and nonempty")

    if shares.shape != (3,) or not np.isclose(shares.sum(), 1) or not (shares > 0).all():
        raise ValueError("fractions must be three positive shares summing to one")

    quotas = shares * len(ids)
    counts = np.floor(quotas).astype(int)
    remainder = len(ids) - counts.sum()
    counts[np.argsort(counts - quotas, kind="stable")[:remainder]] += 1

    if not counts.all():
        raise ValueError("scenario count and fractions must populate all three splits")

    shuffled = np.random.default_rng(seed).permutation(ids).tolist()

    return dict(zip(shuffled, np.repeat(SPLITS, counts).tolist()))


def validate_splits(records: Sequence[PeriodRecord], splits: Mapping[str, str],
                    excluded_ids: Sequence[str]) -> None:
    """Check scenario membership, labels and isolation from customer learning."""
    if set(splits) != {row.scenario_id for row in records} or set(splits.values()) != set(SPLITS):
        raise ValueError("splits must assign every recorded scenario to train, validation or test")

    if set(splits).intersection(excluded_ids):
        raise ValueError("management splits overlap excluded customer scenario IDs")
