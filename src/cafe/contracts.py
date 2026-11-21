"""Shared records, field orders, and catalogue and history operations."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import Config

Array = np.ndarray
PRODUCTS = ("espresso", "latte", "pastry")
ACTION_NAMES = (*(f"price_{p}" for p in PRODUCTS), "staff", *(f"stock_{p}" for p in PRODUCTS))
TARGET_NAMES = (*(f"sales_{p}" for p in PRODUCTS), "abandonment", "mean_wait")
FEATURE_NAMES = (
    "time_sin", "time_cos", "previous_market_arrivals", "previous_A_entries",
    *(f"previous_A_sales_{p}" for p in PRODUCTS), "previous_A_abandonment",
    "previous_A_mean_wait", "wait_valid", *(f"previous_A_waste_{p}" for p in PRODUCTS),
    "current_queue_A", "current_queue_B",
    *(f"previous_A_{field}" for field in ACTION_NAMES),
    *(f"B_{field}" for field in ACTION_NAMES), "observation_valid",
)


@dataclass(frozen=True)
class Action:
    """One catalogue entry in stable ID order."""

    action_id: int
    prices: tuple[float, ...]
    staff: int
    stock: tuple[int, ...]


@dataclass(frozen=True)
class Arrival:
    """One scheduled customer with a policy-independent tie seed."""

    customer_id: int
    minute: int
    type_id: int
    tie_seed: int


@dataclass(frozen=True)
class Scenario:
    """Ordered, exogenous arrivals for one independent day."""

    scenario_id: str
    group: str
    seed: int
    arrivals: tuple[Arrival, ...]


@dataclass(frozen=True)
class CustomerView:
    """Customer-visible state; prices and availability have shape [2, 3]."""

    phase: str
    type_id: int
    price_profile_ids: tuple[int, int]
    prices: Array
    queue_lengths: tuple[int, int]
    availability: Array
    cafe_id: int | None
    product_id: int | None
    queue_position: int
    elapsed_wait: int
    remaining_patience: int
    final_minute: bool


@dataclass(frozen=True)
class Observation:
    """A real pre-decision observation in FEATURE_NAMES order."""

    period_index: int
    values: Array

    def __post_init__(self) -> None:
        """Detach the values from simulator state and mark this snapshot read-only."""
        values = np.array(self.values, dtype=float, copy=True)

        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("observation values must be a finite vector of length 30")

        if self.period_index < 0 or values[-1] != 1:
            raise ValueError("real observations require period_index >= 0 and observation_valid=1")

        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class PeriodRecord:
    """Factual action and accounting for café A in one period."""

    scenario_id: str
    trajectory_id: str
    period_index: int
    observation: Observation
    action: Action
    sales: Array
    entries: int
    abandonment: int
    mean_wait: float
    wait_valid: bool
    waste: Array
    revenue: float
    preparation_cost: float
    staff_cost: float
    profit: float
    diagnostics: dict


@dataclass(frozen=True)
class DayResult:
    """Period records and diagnostics for a single rollout."""

    scenario_id: str
    trajectory_id: str
    periods: list[PeriodRecord]
    diagnostics: dict


@dataclass(frozen=True)
class DecisionInfo:
    """Diagnostics for the last manager decision."""

    action_id: int
    predicted_outcomes: Array | None
    predicted_profit: float | None
    predicted_feasible: bool | None
    fallback_used: bool


@dataclass(frozen=True)
class DatasetArrays:
    """Raw-unit arrays and row keys for one dataset split."""

    history: Array
    actions: Array
    targets: Array
    target_mask: Array
    keys: Array


@dataclass(frozen=True)
class ScalerStats:
    """Training-only scaling statistics and their source scenario IDs."""

    observation_mean: Array
    observation_std: Array
    action_mean: Array
    action_std: Array
    target_std: Array
    training_scenario_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactManifest:
    """Artifact identity, field schema and upstream provenance."""

    schema_version: str
    simulator_version: str
    config_hash: str
    catalogue_hash: str
    parent_hashes: dict[str, str]
    scenario_ids: tuple[str, ...]
    seeds: dict[str, int]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    package_versions: dict[str, str]


def build_catalogue(config: Config) -> tuple[Action, ...]:
    """Enumerate price, staff and stock profiles, varying stock fastest."""
    cat = config.catalogue
    combinations = product(cat["price_multipliers"], cat["staff_counts"], cat["stock_profiles"])

    return tuple(
        Action(index, tuple(round(p["base_price"] * multiplier, 2) for p in config.products),
               staff, tuple(stock))
        for index, (multiplier, staff, stock) in enumerate(combinations)
    )


def validate_action(action: Action, catalogue: Sequence[Action]) -> None:
    """Check the complete action against the canonical entry for its ID."""
    index = action.action_id

    if type(index) is not int or not 0 <= index < len(catalogue) or catalogue[index] != action:
        raise ValueError(f"action conflicts with catalogue: {action!r}")


def make_history(observations: Sequence[Observation], H: int) -> Array:
    """Copy the last H real rows from one trajectory into a zero-left-padded window."""
    if type(H) is not int or H < 1:
        raise ValueError("H must be a positive integer")

    rows = observations[-H:]
    window = np.zeros((H, len(FEATURE_NAMES)))

    for offset, row in enumerate(rows):
        if offset and row.period_index != rows[offset - 1].period_index + 1:
            raise ValueError("history period indices must be consecutive within one day")

        window[H - len(rows) + offset] = row.values

    return window
