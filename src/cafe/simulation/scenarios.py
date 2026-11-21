"""Generate and persist policy-independent arrival schedules and decision streams."""

from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.random import Generator

from ..config import Config, validate_config
from ..contracts import Arrival, Scenario


def _validate_scenario(scenario: Scenario) -> None:
    """Reject malformed identities, duplicate customers and unordered arrivals."""
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id:
        raise ValueError("scenario_id must be a nonempty string")

    if not isinstance(scenario.group, str) or not scenario.group:
        raise ValueError("scenario group must be a nonempty string")

    if type(scenario.seed) is not int or not 0 <= scenario.seed <= np.iinfo(np.int64).max:
        raise ValueError("scenario seed must be a nonnegative int64 integer")

    seen = set()
    previous_minute = -1

    for arrival in scenario.arrivals:
        for field in ("customer_id", "minute", "type_id", "tie_seed"):
            value = getattr(arrival, field)

            if type(value) is not int or not 0 <= value <= np.iinfo(np.int64).max:
                raise ValueError(f"arrival {field} must be a nonnegative int64 integer")

        if arrival.customer_id in seen or arrival.minute < previous_minute:
            raise ValueError("arrivals require unique customer_id values and ordered minutes")

        seen.add(arrival.customer_id)
        previous_minute = arrival.minute


def generate_scenario(config: Config, scenario_id: str, group: str, seed: int) -> Scenario:
    """Sample the specification's daily multiplier, stationary AR(1) demand and arrivals."""
    validate_config(config)
    _validate_scenario(Scenario(scenario_id, group, seed, ()))
    rng = np.random.default_rng(seed)
    demand = config.demand
    persistence = demand["latent_persistence"]
    noise_std = demand["latent_noise_std"]
    log_bounds = np.log(demand["latent_multiplier"])
    day_multiplier = rng.uniform(*demand["daily_multiplier"])
    latent = rng.normal(0, noise_std / np.sqrt(1 - persistence ** 2))
    arrivals = []
    probabilities = [customer["probability"] for customer in config.customer_types]

    for minute in range(config.simulation["day_minutes"]):
        if minute and minute % config.simulation["period_minutes"] == 0:
            latent = persistence * latent + rng.normal(0, noise_std)

        base_rate = demand["base_rate"]

        for peak in demand["peaks"]:
            offset = (minute - peak["center_minute"]) / peak["width_minutes"]
            base_rate += peak["amplitude"] * np.exp(-0.5 * offset ** 2)

        rate = base_rate * day_multiplier * np.exp(np.clip(latent, *log_bounds))

        for _ in range(rng.poisson(rate)):
            arrivals.append(Arrival(len(arrivals), minute,
                                    int(rng.choice(len(probabilities), p=probabilities)),
                                    int(rng.integers(0, np.iinfo(np.int64).max))))

    return Scenario(scenario_id, group, seed, tuple(arrivals))


def save_scenario(scenario: Scenario, path: Path) -> None:
    """Write an NPZ schedule without pickle, preserving same-minute arrival order."""
    _validate_scenario(scenario)
    rows = np.array([(a.customer_id, a.minute, a.type_id, a.tie_seed)
                     for a in scenario.arrivals], dtype=np.int64).reshape(-1, 4)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as stream:
        np.savez_compressed(stream, scenario_id=scenario.scenario_id, group=scenario.group,
                            seed=np.int64(scenario.seed), arrivals=rows)


def load_scenario(path: Path) -> Scenario:
    """Restore an immutable schedule; reject malformed NPZ fields with the file name."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            rows = saved["arrivals"]
            seed = saved["seed"]
            identity = saved["scenario_id"]
            group = saved["group"]

            if rows.ndim != 2 or rows.shape[1] != 4 or rows.dtype.kind not in "iu":
                raise ValueError("arrivals must be an integer array with shape [N,4]")

            if seed.shape != () or seed.dtype.kind not in "iu":
                raise ValueError("seed must be an integer scalar")

            if any(value.shape != () or value.dtype.kind != "U" for value in (identity, group)):
                raise ValueError("scenario_id and group must be string scalars")

            scenario = Scenario(str(identity.item()), str(group.item()), int(seed),
                                tuple(Arrival(*(int(v) for v in row)) for row in rows))

        _validate_scenario(scenario)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid scenario file {path}: {error}") from error

    return scenario


def customer_rng(scenario: Scenario, customer_id: int, minute: int, phase: str) -> Generator:
    """Key a decision stream so extra draws never perturb another customer or context."""
    if phase not in ("arrival", "waiting") or type(minute) is not int or minute < 0:
        raise ValueError("customer RNG requires phase arrival/waiting and nonnegative minute")

    if type(customer_id) is not int or customer_id < 0:
        raise ValueError("customer_id must be a nonnegative integer")

    # Generated schedules use contiguous IDs; arbitrary saved IDs still retain their order.
    arrival = scenario.arrivals[customer_id] if customer_id < len(scenario.arrivals) else None

    if arrival is None or arrival.customer_id != customer_id:
        arrival = next((a for a in scenario.arrivals if a.customer_id == customer_id), None)

    if arrival is None:
        raise ValueError(f"customer_id {customer_id!r} is absent from scenario")

    key = repr((scenario.scenario_id, scenario.seed, arrival.tie_seed, customer_id, minute, phase))
    seed = int.from_bytes(sha256(key.encode()).digest(), "little")

    return np.random.default_rng(seed)
