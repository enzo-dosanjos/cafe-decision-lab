"""Encode customer observations and learn masked tabular policies without simulation."""

from bisect import bisect_right
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.random import Generator

from cafe.config import Config
from cafe.contracts import ArtifactManifest, CustomerView

StateKey = tuple[str | int, ...]
STATE_SCHEMA = "customer-state-v1"


@dataclass(frozen=True)
class PendingTransition:
    """Customer type, phase-tagged state and action awaiting a reward."""

    type_id: int
    state: StateKey
    action_id: int


def _integer(value: object, field: str, minimum: int = 0) -> None:
    """Require a nonnegative integer identifier or count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{field}: expected an integer >= {minimum}")


def _validate_view(view: CustomerView) -> None:
    """Reject malformed customer-visible arrays, identifiers and minute counts."""
    if view.phase not in ("arrival", "waiting"):
        raise ValueError("phase: expected arrival or waiting")

    prices = np.asarray(view.prices)
    availability = np.asarray(view.availability)

    if prices.shape != (2, 3) or not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("prices: expected finite nonnegative array with shape [2,3]")

    if availability.shape != (2, 3) or not np.isin(availability, (0, 1)).all():
        raise ValueError("availability: expected boolean array with shape [2,3]")

    for field in ("price_profile_ids", "queue_lengths"):
        values = getattr(view, field)

        if len(values) != 2:
            raise ValueError(f"{field}: expected two integers")

        for value in values:
            _integer(value, field)

    for field in ("type_id", "queue_position", "elapsed_wait", "remaining_patience"):
        _integer(getattr(view, field), field)

    if not isinstance(view.final_minute, (bool, np.bool_)):
        raise ValueError("final_minute: expected a boolean")

    if view.phase == "waiting":
        for field, limit in (("cafe_id", 2), ("product_id", 3)):
            value = getattr(view, field)
            _integer(value, field)

            if value >= limit:
                raise ValueError(f"{field}: expected an index below {limit}")


def encode_state(view: CustomerView) -> StateKey:
    """Bin queues, elapsed minutes and patience; availability bits follow product IDs.

    Exhausted patience uses the lowest patience bin, since its mask forces leave.
    Type selects a separate table and is never included in the state key.
    """
    _validate_view(view)

    if view.phase == "arrival":
        masks = tuple(sum(int(flag) << product for product, flag in enumerate(row))
                      for row in view.availability)

        return ("arrival", int(view.price_profile_ids[0]),
                *(bisect_right((1, 4, 8), length) for length in view.queue_lengths), *masks)

    cafe, product = view.cafe_id, view.product_id

    return ("waiting", int(cafe), int(product), int(view.price_profile_ids[cafe]),
            bisect_right((1, 4, 8), view.queue_position),
            bisect_right((3, 6, 11), view.elapsed_wait),
            bisect_right((3, 6), view.remaining_patience),
            int(view.availability[cafe][product]), int(view.final_minute))


def valid_actions(view: CustomerView, config: Config) -> tuple[int, ...]:
    """Return affordable stocked arrivals plus outside, or wait/forced leave IDs."""
    _validate_view(view)

    if view.type_id >= len(config.customer_types):
        raise ValueError("type_id: outside configured customer_types")

    if any(profile >= len(config.catalogue["price_multipliers"]) for profile in view.price_profile_ids):
        raise ValueError("price_profile_ids: outside configured price_multipliers")

    customer = config.customer_types[view.type_id]
    affordable = np.asarray(view.prices) <= customer["budget"]
    available = np.asarray(view.availability, dtype=bool) & affordable

    if view.phase == "arrival":
        return (*map(int, np.flatnonzero(available)), 6)

    forced = (not available[view.cafe_id, view.product_id]
              or view.remaining_patience == 0 or view.elapsed_wait >= customer["max_wait"]
              or view.final_minute)

    return (1,) if forced else (0, 1)


def _validate_state(state: StateKey) -> int:
    """Validate phase-specific key dimensions and bins; return the action count."""
    if not isinstance(state, tuple) or not state or state[0] not in ("arrival", "waiting"):
        raise ValueError("state: expected an arrival or waiting tuple")

    bounds = (None, 3, 3, 7, 7) if state[0] == "arrival" else (1, 2, None, 3, 3, 2, 1, 1)

    if len(state) != len(bounds) + 1:
        raise ValueError(f"state: {state[0]} requires {len(bounds)} discrete fields")

    for value, maximum in zip(state[1:], bounds):
        _integer(value, "state field")

        if maximum is not None and value > maximum:
            raise ValueError(f"state field: expected a bin <= {maximum}")

    return 7 if state[0] == "arrival" else 2


def _actions(state: StateKey, actions: Sequence[int]) -> tuple[int, ...]:
    """Validate a nonempty mask with distinct phase-appropriate action IDs."""
    count = _validate_state(state)
    result = tuple(actions)

    if not result:
        raise ValueError("valid_actions: expected at least one action")

    for action in result:
        _integer(action, "action_id")

        if action >= count:
            raise ValueError(f"action_id: expected an index below {count}")

    if len(set(result)) != len(result):
        raise ValueError("valid_actions: expected distinct actions")

    return result


def _probability(value: float, field: str) -> None:
    """Require a finite learning parameter between zero and one."""
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field}: expected a finite number in [0,1]")


class CustomerPolicy:
    """Sparse tables indexed by type, phase-tagged state, then action ID.

    Each row is a list of seven arrival or two waiting Q values. Missing rows
    read as zeros. Constructor inputs and frozen snapshots do not share storage.
    """

    def __init__(self, tables: dict | None = None, *, frozen: bool = False) -> None:
        """Copy and validate Q tables; frozen policies reject learning/exploration."""
        self._tables = {}
        self._frozen = frozen

        if not isinstance(frozen, bool):
            raise ValueError("frozen: expected a boolean")

        for type_id, states in ({} if tables is None else tables).items():
            _integer(type_id, "type_id")
            self._tables[int(type_id)] = {}

            for state, values in states.items():
                count = _validate_state(state)
                row = np.asarray(values, dtype=float)

                if row.shape != (count,) or not np.isfinite(row).all():
                    raise ValueError(f"Q values: expected {count} finite values for {state}")

                self._tables[int(type_id)][state] = row.tolist()

    @property
    def frozen(self) -> bool:
        """Report whether learning and exploration are disabled."""
        return self._frozen

    @property
    def tables(self) -> dict:
        """Expose mutable training tables or a detached copy of frozen tables."""
        return deepcopy(self._tables) if self.frozen else self._tables

    def choose(self, type_id: int, state: StateKey, valid_actions: Sequence[int],
               *, epsilon: float, rng: Generator) -> int:
        """Choose uniformly among exploratory actions or tied valid maxima."""
        _integer(type_id, "type_id")
        actions = _actions(state, valid_actions)
        _probability(epsilon, "epsilon")

        if self.frozen and epsilon != 0:
            raise ValueError("epsilon: frozen policy requires zero exploration")

        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.choice(actions))

        row = self._tables.get(type_id, {}).get(state, [0.0] * _validate_state(state))
        best = max(row[action] for action in actions)
        tied = [action for action in actions if row[action] == best]

        return int(rng.choice(tied))

    def update(self, pending: PendingTransition, reward: float, next_state: StateKey | None,
               next_valid_actions: Sequence[int], *, terminal: bool, alpha: float, gamma: float) -> None:
        """Apply one masked Q update; reward is this transition's utility in HKD."""
        if self.frozen:
            raise ValueError("cannot update a frozen customer policy")

        _integer(pending.type_id, "type_id")
        _actions(pending.state, (pending.action_id,))
        _probability(alpha, "alpha")
        _probability(gamma, "gamma")

        if not isfinite(reward):
            raise ValueError("reward: expected a finite number")

        target = reward

        if not terminal:
            actions = _actions(next_state, next_valid_actions)
            next_row = self._tables.get(pending.type_id, {}).get(next_state, [0.0] * _validate_state(next_state))
            target += gamma * max(next_row[action] for action in actions)

        states = self._tables.setdefault(pending.type_id, {})
        row = states.setdefault(pending.state, [0.0] * _validate_state(pending.state))
        row[pending.action_id] += alpha * (target - row[pending.action_id])

    def freeze(self) -> "CustomerPolicy":
        """Return an independent snapshot that rejects Q updates."""
        return CustomerPolicy(self._tables, frozen=True)

    def fingerprint(self) -> str:
        """Hash sorted tables and the state schema, excluding mutability and lineage."""
        payload = {"state_schema": STATE_SCHEMA, "tables": _entries(self)}

        return sha256(_json(payload).encode()).hexdigest()


def _entries(policy: CustomerPolicy) -> list[dict]:
    """Encode integer type IDs and tuple keys explicitly in stable JSON order."""
    return [{"type_id": type_id, "states": [
        {"state": list(state), "values": values} for state, values in sorted(states.items())]}
        for type_id, states in sorted(policy._tables.items())]


def _json(value: object) -> str:
    """Serialize deterministic JSON, rejecting nonfinite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save_policy(policy: CustomerPolicy, path: Path, manifest: ArtifactManifest) -> None:
    """Write policy JSON with explicit keys, schema, fingerprint and full lineage."""
    payload = {"state_schema": STATE_SCHEMA, "manifest": asdict(manifest),
               "tables": _entries(policy), "fingerprint": policy.fingerprint()}
    encoded = _json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def load_policy(path: Path, expected: ArtifactManifest) -> CustomerPolicy:
    """Validate JSON schema, full manifest and contents; restore a frozen policy."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload["state_schema"] != STATE_SCHEMA:
            raise ValueError("state_schema: incompatible customer state schema")

        manifest = payload["manifest"]

        for field, value in json.loads(_json(asdict(expected))).items():
            if manifest.get(field) != value:
                raise ValueError(f"manifest.{field}: incompatible artifact lineage")

        tables = {}

        for entry in payload["tables"]:
            type_id = entry["type_id"]
            _integer(type_id, "type_id")

            if type_id in tables:
                raise ValueError("tables: duplicate type_id")

            tables[type_id] = {}

            for row in entry["states"]:
                state = tuple(row["state"])
                _validate_state(state)

                if state in tables[type_id]:
                    raise ValueError("tables: duplicate state")

                tables[type_id][state] = row["values"]

        policy = CustomerPolicy(tables, frozen=True)

        if policy.fingerprint() != payload["fingerprint"]:
            raise ValueError("fingerprint: policy contents do not match")

    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Invalid customer policy in {path}: {error}") from error

    return policy
