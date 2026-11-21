"""Load experiment settings and identify the resolved configuration reproducibly."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isclose, isfinite, prod
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Resolved experiment settings, with product and type order preserved."""

    simulation: dict
    demand: dict
    products: tuple[dict, ...]
    customer_types: tuple[dict, ...]
    catalogue: dict
    customer_training: dict
    dataset: dict
    model: dict
    training: dict
    evaluation: dict
    seeds: dict[str, int]


def load_config(path: Path) -> Config:
    """Load JSON settings, overlaying section keys on an adjacent default.json."""
    with path.open() as stream:
        settings = json.load(stream)

    defaults_path = path.with_name("default.json")

    if path != defaults_path and defaults_path.exists():
        with defaults_path.open() as stream:
            defaults = json.load(stream)

        for key, value in settings.items():
            if isinstance(defaults.get(key), dict) and isinstance(value, dict):
                defaults[key].update(value)
            else:
                defaults[key] = value

        settings = defaults

    try:
        for key in ("products", "customer_types"):
            settings[key] = tuple(settings[key])

        config = Config(**settings)
        validate_config(config)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid configuration in {path}: {error}") from error

    return config


def _number(value: object, field: str, *, integer: bool = False, positive: bool = False) -> None:
    """Require a finite nonnegative number, optionally integral or strictly positive."""
    allowed = (int,) if integer else (int, float)

    if type(value) not in allowed or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{field}: expected a {'positive' if positive else 'nonnegative'} "
                         f"{'integer' if integer else 'finite number'}, got {value!r}")


def validate_config(config: Config) -> None:
    """Reject inconsistent time units, dimensions, probabilities and experiment settings."""
    demand = config.demand
    _number(demand["base_rate"], "demand.base_rate")
    _number(demand["latent_noise_std"], "demand.latent_noise_std")
    persistence = demand["latent_persistence"]

    if type(persistence) not in (int, float) or not -1 < persistence < 1:
        raise ValueError("demand.latent_persistence must lie strictly between -1 and 1")

    for field in ("daily_multiplier", "latent_multiplier"):
        bounds = demand[field]

        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"demand.{field} requires two bounds")

        for bound in bounds:
            _number(bound, f"demand.{field}", positive=field == "latent_multiplier")

        if bounds[0] > bounds[1]:
            raise ValueError(f"demand.{field} requires ordered bounds")

    if not isinstance(demand["peaks"], list):
        raise ValueError("demand.peaks must be a list")

    for peak in demand["peaks"]:
        for field in ("center_minute", "width_minutes", "amplitude"):
            if not isinstance(peak, dict) or field not in peak:
                raise ValueError(f"demand.peaks requires {field}")

            _number(peak[field], f"demand.peaks.{field}", positive=field == "width_minutes")

    sim = config.simulation

    for field in ("day_minutes", "period_minutes", "service_effort"):
        _number(sim[field], field, integer=True, positive=True)

    if sim["day_minutes"] % sim["period_minutes"]:
        raise ValueError("day_minutes must be divisible by period_minutes")

    _number(sim["staff_period_cost"], "staff_period_cost")

    if tuple(p["name"] for p in config.products) != ("espresso", "latte", "pastry"):
        raise ValueError("products must be ordered espresso, latte, pastry")

    for product in config.products:
        for field in ("base_price", "unit_cost"):
            _number(product[field], field)

    for customer in config.customer_types:
        if len(customer["values"]) != 3:
            raise ValueError("customer values must contain three product utilities")

        for field in ("budget", "waiting_cost", "probability"):
            _number(customer[field], field)

        for value in customer["values"]:
            _number(value, "customer product value")

        _number(customer["max_wait"], "max_wait", integer=True, positive=True)

    if not isclose(sum(c["probability"] for c in config.customer_types), 1):
        raise ValueError("customer probabilities must sum to one")

    cat = config.catalogue

    for field in ("price_multipliers", "staff_counts", "stock_profiles"):
        values = cat[field]

        if not values or len({json.dumps(v) for v in values}) != len(values):
            raise ValueError(f"{field} must contain distinct profiles")

        for value in values:
            if field == "stock_profiles":
                if len(value) != 3:
                    raise ValueError("stock_profiles must have three quantities")

                for qty in value:
                    _number(qty, field, integer=True)
            else:
                _number(value, field, integer=field == "staff_counts", positive=True)

    _number(sim["competitor_action_id"], "competitor_action_id", integer=True)
    if sim["competitor_action_id"] >= prod(len(cat[k]) for k in ("price_multipliers", "staff_counts", "stock_profiles")):
        raise ValueError("competitor_action_id outside catalogue")

    for section, fields in (
        (config.customer_training, ("days", "validation_interval", "validation_days")),
        (config.dataset, ("scenario_days", "trajectories_per_day")),
        (config.training, ("max_epochs", "batch_size", "patience")),
        (config.model, ("history_length",)),
    ):
        for field in fields:
            _number(section[field], field, integer=True, positive=True)

    if config.model["feature_count"] != 30:
        raise ValueError("feature_count must equal 30")

    fractions = config.dataset["fractions"]

    if len(fractions) != 3 or not isclose(sum(fractions), 1):
        raise ValueError("dataset fractions must contain three shares summing to one")

    for value in fractions:
        _number(value, "dataset fraction", positive=True)

    for field in ("alpha", "gamma", "epsilon_start", "epsilon_end"):
        value = config.customer_training[field]
        _number(value, field)
        if value > 1:
            raise ValueError(f"{field} must be at most one")

    for field in ("learning_rate", "gradient_clip"):
        _number(config.training[field], field, positive=True)

    _number(config.evaluation["abandonment_limit"], "abandonment_limit")
    for field in ("scenario", "management", "customer", "split", "training"):
        _number(config.seeds[field], field, integer=True)


def config_hash(config: Config) -> str:
    """Hash the canonical JSON of the resolved settings, independent of file formatting."""
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)

    return sha256(encoded.encode()).hexdigest()
