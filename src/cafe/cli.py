"""Run café experiment stages and reproducible integration checks."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import numpy as np

from .config import Config, config_hash, load_config
from .contracts import ArtifactManifest, TARGET_NAMES, FEATURE_NAMES, Observation, Scenario, build_catalogue, make_history, validate_action
from .managers.fixed import FixedManager


def _save_report(report: dict, path: Path) -> None:
    """Replace a smoke report only after all checks and JSON encoding succeed."""
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(encoded + "\n")

    Path(stream.name).replace(path)


def smoke(config: Config, output_dir: Path) -> dict:
    """Exercise the shared observation/action boundary and save a deterministic fixture."""
    catalogue = build_catalogue(config)
    action = catalogue[config.simulation["competitor_action_id"]]
    manager = FixedManager(action)
    manager.reset()

    # Opening fixture: previous metrics and queues are zero, and B's settings are known.
    values = np.zeros(len(FEATURE_NAMES))
    values[FEATURE_NAMES.index("time_cos")] = 1
    values[-8:-1] = (*action.prices, action.staff, *action.stock)
    values[-1] = 1
    history = make_history([Observation(0, values)], config.model["history_length"])
    chosen = manager.choose(history)
    validate_action(chosen, catalogue)

    report = {
        "stage": "contracts", "config_hash": config_hash(config),
        "feature_names": FEATURE_NAMES, "opening_history": history.tolist(),
        "catalogue": [asdict(entry) for entry in catalogue],
        "selected_action_id": chosen.action_id,
    }
    _save_report(report, output_dir / "contracts.json")

    return report


def _rollout(config: Config, output_dir: Path) -> dict:
    """Run the integrated simulator on an empty schedule and check every period."""
    from .customers.policy import CustomerPolicy
    from .simulation.simulator import run_day

    scenario = Scenario("smoke-empty", "smoke", config.seeds["scenario"], ())
    action = build_catalogue(config)[config.simulation["competitor_action_id"]]
    policy = CustomerPolicy(frozen=True)
    original_hash = policy.fingerprint()
    result = run_day(scenario, policy, FixedManager(action), config, trajectory_id="fixed")

    count = config.simulation["day_minutes"] // config.simulation["period_minutes"]
    prep_cost = sum(q * p["unit_cost"] for q, p in zip(action.stock, config.products))
    staff_cost = action.staff * config.simulation["staff_period_cost"]

    if len(result.periods) != count or policy.fingerprint() != original_hash:
        raise ValueError("empty day must preserve the policy and produce one record per period")

    rows = []

    for index, row in enumerate(result.periods):
        if (row.period_index != index or row.action != action or row.entries or row.abandonment
                or row.revenue or row.wait_valid or row.mean_wait or np.any(row.sales)
                or not np.array_equal(row.waste, action.stock)
                or not isclose(row.preparation_cost, prep_cost)
                or not isclose(row.staff_cost, staff_cost)
                or not isclose(row.profit, -prep_cost - staff_cost)):
            raise ValueError(f"zero-arrival accounting mismatch in period {index}")

        rows.append({"period_index": index, "action_id": action.action_id,
                     "profit": float(row.profit), "waste": row.waste.tolist()})

    report = {"stage": "simulation", "scenario_id": scenario.scenario_id,
              "config_hash": config_hash(config), "periods": rows}
    _save_report(report, output_dir / "day.json")

    return report


def _train_customers(config: Config, output_dir: Path) -> None:
    """Create separate scenario groups, record lineage and run customer training."""
    from .customers.training import train_customers
    from .simulation.scenarios import generate_scenario, save_scenario

    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        raise ValueError(f"Customer training needs an empty experiment directory: {output_dir}")

    catalogue = build_catalogue(config)
    catalogue_rows = [asdict(action) for action in catalogue]
    catalogue_json = json.dumps(catalogue_rows, sort_keys=True, separators=(",", ":"))
    (output_dir / "catalogue.json").write_text(catalogue_json + "\n", encoding="utf-8")
    _save_report(asdict(config), output_dir / "config.json")

    # Namespace customer schedule seeds separately from management experiments.
    rng = np.random.default_rng([config.seeds["scenario"], 0x43555354])
    groups = {}
    schedule_rows = []
    settings = config.customer_training

    for group, count in (("customer-train", settings["days"]),
                         ("customer-validation", settings["validation_days"])):
        groups[group] = []

        for index in range(count):
            seed = int(rng.integers(2**63 - 1))
            scenario = generate_scenario(config, f"{group}-{index:05d}", group, seed)
            groups[group].append(scenario)
            save_scenario(scenario, output_dir / "scenarios" / f"{scenario.scenario_id}.npz")
            schedule_rows.append({"id": scenario.scenario_id, "group": group, "seed": seed})

    _save_report({"scenarios": schedule_rows}, output_dir / "scenarios" / "manifest.json")
    manifest = ArtifactManifest(
        schema_version="1", simulator_version="1", config_hash=config_hash(config),
        catalogue_hash=sha256(catalogue_json.encode()).hexdigest(),
        parent_hashes={"schedules": sha256((output_dir / "scenarios" / "manifest.json").read_bytes()).hexdigest()},
        scenario_ids=tuple(row["id"] for row in schedule_rows), seeds=config.seeds,
        feature_names=FEATURE_NAMES, target_names=TARGET_NAMES,
        package_versions={"numpy": np.__version__, "python": platform.python_version()},
    )
    _save_report(asdict(manifest), output_dir / "manifest.json")

    # Cycle a seeded permutation so every 27 training days covers the full catalogue.
    manager_rng = np.random.default_rng(config.seeds["management"])
    managers = [FixedManager(catalogue[int(i)]) for i in manager_rng.permutation(len(catalogue))]
    train_customers(config, groups["customer-train"], groups["customer-validation"],
                    managers, output_dir / "customers", manifest)
    print(f"Selected frozen policy: {output_dir / 'customers' / 'policy.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch smoke checks with explicit configuration and output paths."""
    parser = argparse.ArgumentParser(prog="python -m cafe.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--config", type=Path, default=Path("config/smoke.json"))
    command.add_argument("--output-dir", type=Path, default=Path("runs/smoke"))
    command.add_argument("--rollout", action="store_true", help="also check the real simulator")
    train = sub.add_parser("train-customers", help="train and select frozen customer Q-tables")
    train.add_argument("--config", type=Path, default=Path("config/default.json"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/customer-training"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.command == "train-customers":
        _train_customers(config, args.output_dir)

        return 0

    smoke(config, args.output_dir)
    print(f"Contract smoke passed: {args.output_dir / 'contracts.json'}")
    if args.rollout:
        try:
            _rollout(config, args.output_dir)
        except ModuleNotFoundError as error:
            if error.name not in {"cafe.simulation", "cafe.simulation.simulator",
                                  "cafe.customers", "cafe.customers.policy"}:
                raise

            parser.exit(2, f"Rollout requires the Simulation and Customers implementations ({error.name}).\n")

        print(f"Simulation smoke passed: {args.output_dir / 'day.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
