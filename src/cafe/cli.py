"""Run café experiment stages and reproducible integration checks."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING
import platform
from time import perf_counter
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np

from .config import Config, config_hash, load_config
from .contracts import Action, ArtifactManifest, TARGET_NAMES, FEATURE_NAMES, Observation, Scenario, build_catalogue, make_history, validate_action
from .managers.fixed import FixedManager

if TYPE_CHECKING:
    from .customers.policy import CustomerPolicy


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


def _catalogue_json(catalogue: Sequence[Action]) -> str:
    """Use the same canonical catalogue bytes for both stages' provenance."""
    rows = [asdict(action) for action in catalogue]

    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _train_customers(config: Config, output_dir: Path) -> None:
    """Create separate scenario groups, record lineage and run customer training."""
    from .customers.training import train_customers
    from .simulation.scenarios import generate_scenario, save_scenario

    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        raise ValueError(f"Customer training needs an empty experiment directory: {output_dir}")

    catalogue = build_catalogue(config)
    catalogue_json = _catalogue_json(catalogue)
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


def _load_dataset_policy(config: Config, customer_dir: Path, catalogue: str
                         ) -> tuple["CustomerPolicy", ArtifactManifest, set[int]]:
    """Verify the selected experiment and collect seeds to exclude from management days."""
    from .customers.policy import load_policy

    # Validate the checkpoint against its training experiment, then check the shared market.
    source = ArtifactManifest(**json.loads((customer_dir / "manifest.json").read_text()))
    trained = load_config(customer_dir / "config.json")
    policy_path = customer_dir / "customers" / "policy.json"
    policy = load_policy(policy_path, source)
    schedule_file = (customer_dir / "scenarios" / "manifest.json").read_bytes()
    source_rows = json.loads(schedule_file)["scenarios"]

    if (source.schema_version != "1" or source.simulator_version != "1"
            or tuple(source.feature_names) != FEATURE_NAMES or tuple(source.target_names) != TARGET_NAMES
            or source.config_hash != config_hash(trained) or config.customer_types != trained.customer_types
            or config.products != trained.products or config.catalogue != trained.catalogue
            or source.catalogue_hash != sha256(catalogue.encode()).hexdigest()
            or source.parent_hashes["schedules"] != sha256(schedule_file).hexdigest()
            or list(source.scenario_ids) != [row["id"] for row in source_rows]):
        raise ValueError(f"{customer_dir}: incompatible customer experiment or market configuration")

    return policy, source, {row["seed"] for row in source_rows}


def _generate_data(config: Config, customer_dir: Path, output_dir: Path) -> None:
    """Publish a complete management dataset from a compatible customer experiment."""
    from .data.generate import build_dataset
    from .data.scaling import fit_scalers, save_scalers
    from .data.splits import SPLITS, make_split_manifest, validate_splits
    from .data.storage import save_arrays, save_manifest, save_records
    from .data.windows import make_windows
    from .simulation.scenarios import generate_scenario

    start = perf_counter()

    if config.dataset["trajectories_per_day"] not in (1, 2):
        raise ValueError("dataset.trajectories_per_day must be 1 (uniform) or 2 (uniform and hold)")

    if output_dir.exists():
        raise FileExistsError(f"Dataset output must be a new directory: {output_dir}")

    catalogue = _catalogue_json(build_catalogue(config))
    policy, source, used = _load_dataset_policy(config, customer_dir, catalogue)
    policy_path = customer_dir / "customers" / "policy.json"
    checkpoint_hash = sha256(policy_path.read_bytes()).hexdigest()

    # Keep management scenarios independent from customer training and exploration.
    rng = np.random.default_rng([config.seeds["scenario"], 0x4D474D54])
    manager_rng = np.random.default_rng(config.seeds["management"])
    scenarios, schedule_rows, exploration = [], [], {}

    for index in range(config.dataset["scenario_days"]):
        seed = int(rng.integers(2**63 - 1))

        while seed in used:
            seed = int(rng.integers(2**63 - 1))

        used.add(seed)
        identity = f"management-{index:05d}"
        scenarios.append(generate_scenario(config, identity, "management", seed))
        exploration[identity] = manager_rng.integers(2**63 - 1, size=config.dataset["trajectories_per_day"]).tolist()
        schedule_rows.append({"id": identity, "group": "management", "seed": seed,
                              "exploration_seeds": exploration[identity]})

    splits = make_split_manifest(list(exploration), seed=config.seeds["split"],
                                 fractions=tuple(config.dataset["fractions"]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # A failed or concurrent run cannot publish a partial dataset.
    with TemporaryDirectory(dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "experiment"
        schedules_path = stage / "scenarios" / "manifest.json"
        _save_report({"scenarios": schedule_rows}, schedules_path)
        parents = {"customer_policy": policy.fingerprint(), "customer_checkpoint": checkpoint_hash,
                   "schedules": sha256(schedules_path.read_bytes()).hexdigest()}
        manifest = ArtifactManifest("1", "1", config_hash(config), source.catalogue_hash, parents,
                                    tuple(exploration), config.seeds, FEATURE_NAMES, TARGET_NAMES,
                                    {"python": platform.python_version(), "numpy": np.__version__})
        _save_report(asdict(config), stage / "config.json")
        (stage / "catalogue.json").write_text(catalogue + "\n", encoding="utf-8")
        records = build_dataset(scenarios, policy, config, exploration, stage, manifest)
        validate_splits(records, splits, source.scenario_ids)
        dataset = stage / "dataset"
        save_manifest(manifest, splits, dataset)
        save_records(records, dataset / "periods.csv", splits=splits)
        save_scalers(fit_scalers(records, splits), dataset / "scalers.json")

        for split in SPLITS:
            arrays = make_windows(records, splits, split=split, H=config.model["history_length"])
            save_arrays(arrays, dataset / f"{split}.npz")

        _save_report(asdict(manifest), stage / "manifest.json")
        _save_report({"dataset_generation_seconds": perf_counter() - start}, stage / "timing.json")
        stage.rename(output_dir)

    print(f"Saved {len(records)} management periods to {output_dir / 'dataset'}")


def _model_inputs(experiment: Path) -> tuple[Config, ArtifactManifest, dict[str, str]]:
    """Check the dataset identity and bind model lineage to the exact fitting inputs."""
    from .data.storage import load_manifest

    config = load_config(experiment / "config.json")
    root = ArtifactManifest(**json.loads((experiment / "manifest.json").read_text()))
    dataset = experiment / "dataset"
    _, splits = load_manifest(dataset, root)
    catalogue = _catalogue_json(build_catalogue(config))

    if (root.config_hash != config_hash(config) or root.catalogue_hash != sha256(catalogue.encode()).hexdigest()
            or tuple(root.feature_names) != FEATURE_NAMES or tuple(root.target_names) != TARGET_NAMES
            or root.schema_version != "1" or root.simulator_version != "1"):
        raise ValueError(f"{experiment}: incompatible dataset configuration, catalogue or fields")

    parents = dict(root.parent_hashes)

    for filename in ("manifest.json", "splits.json", "train.npz", "validation.npz", "scalers.json"):
        parents[filename] = sha256((dataset / filename).read_bytes()).hexdigest()

    return config, replace(root, parent_hashes=parents), splits


def _train_model(args: argparse.Namespace) -> None:
    """Fit on training periods and select the checkpoint on validation periods."""
    from .data.scaling import load_scalers
    from .data.storage import load_arrays
    from .models.training import train_predictor

    config, manifest, splits = _model_inputs(args.experiment_dir)
    dataset = args.experiment_dir / "dataset"
    arrays = {}
    periods = config.simulation["day_minutes"] // config.simulation["period_minutes"]
    modes = ("uniform", "hold")[:config.dataset["trajectories_per_day"]]

    for split in ("train", "validation"):
        data = load_arrays(dataset / f"{split}.npz", H=config.model["history_length"], F=config.model["feature_count"])
        expected = {(s, mode, i) for s in splits if splits[s] == split for mode in modes for i in range(periods)}

        if set(data.keys.tolist()) != expected:
            raise ValueError(f"{dataset}: {split} array keys disagree with the scenario split")

        arrays[split] = data

    ids = [s for s in splits if splits[s] == "train"]
    scalers = load_scalers(dataset / "scalers.json", ids)
    output = args.experiment_dir / "models" / args.model
    model = train_predictor(args.model, arrays["train"], arrays["validation"], scalers, config,
                            seed=args.seed, device=args.device, output_dir=output, manifest=manifest)
    print(f"Saved {output / f'seed-{args.seed}.pt'} ({sum(p.numel() for p in model.parameters())} parameters)")


def _management_inputs(experiment: Path, customer_dir: Path, split: str) -> tuple:
    """Validate customers and saved schedules before complete management rollouts."""
    from .simulation.scenarios import load_scenario

    config, manifest, splits = _model_inputs(experiment)
    catalogue = build_catalogue(config)
    policy, _, excluded_seeds = _load_dataset_policy(config, customer_dir, _catalogue_json(catalogue))
    policy_path = customer_dir / "customers" / "policy.json"

    if (policy.fingerprint() != manifest.parent_hashes["customer_policy"]
            or sha256(policy_path.read_bytes()).hexdigest() != manifest.parent_hashes["customer_checkpoint"]):
        raise ValueError("selected frozen customers do not match the dataset")

    schedule_manifest = (experiment / "scenarios" / "manifest.json").read_bytes()

    if sha256(schedule_manifest).hexdigest() != manifest.parent_hashes["schedules"]:
        raise ValueError("incompatible management schedule manifest")

    seeds = {r["id"]: r["seed"] for r in json.loads(schedule_manifest)["scenarios"]}
    scenarios = []

    for identity in sorted(s for s in splits if splits[s] == split):
        scenario = load_scenario(experiment / "scenarios" / f"{identity}.npz")

        if (scenario.scenario_id != identity or scenario.group != "management"
                or scenario.seed != seeds[identity] or scenario.seed in excluded_seeds):
            raise ValueError(f"incompatible management schedule {identity}")

        scenarios.append(scenario)

    return config, manifest, catalogue, policy, scenarios


def _validate_managers(args: argparse.Namespace) -> None:
    """Rank baseline and neural candidates on saved validation days only."""
    from itertools import product

    from .evaluate import select_managers
    from .managers.adaptive import AdaptiveManager
    from .managers.mlp import MLPManager
    from .models.training import load_checkpoint

    experiment = args.experiment_dir
    output = args.output_dir

    if output.exists():
        raise FileExistsError(output)

    config, manifest, catalogue, policy, scenarios = _management_inputs(experiment, args.customer_dir, "validation")
    seeds = args.seeds if args.seeds is not None else [args.seed]

    if len(seeds) != len(set(seeds)) or min(seeds) < 0 or len(args.models) != len(set(args.models)):
        raise ValueError("expected distinct nonnegative seeds and distinct model families")

    variants = [{"alpha": a, "buffer": b, "queue_threshold": q}
                for a, b, q in product((0.3, 0.6), (1.0, 1.25), (3, 6))]
    settings = {"fixed": [{"action_id": a.action_id} for a in catalogue], "adaptive": variants}
    candidates = {"fixed": [FixedManager(a) for a in catalogue],
                  "adaptive": [AdaptiveManager(catalogue, config, **v) for v in variants]}
    cached_trials = []

    if args.baseline_selection:
        cached = json.loads(args.baseline_selection.read_text())

        if cached["manifest"] != json.loads(json.dumps(asdict(manifest))):
            raise ValueError("baseline selection belongs to another dataset")

        for family in ("fixed", "adaptive"):
            rows = [r for r in cached["trials"] if r["family"] == family]

            if (len(rows) != len(settings[family])
                    or {r["candidate"] for r in rows} != set(range(len(settings[family])))
                    or any(r["settings"] != settings[family][r["candidate"]]
                           or {d["scenario_id"] for d in r["days"]} != {d.scenario_id for d in scenarios}
                           or len(r["days"]) != len(scenarios) for r in rows)):
                raise ValueError("baseline selection has incompatible candidates or validation days")

            cached_trials.extend(rows)
            del candidates[family]

    from .managers.lstm import LSTMManager

    for family in args.models:
        settings[family], candidates[family] = [], []

        for seed in seeds:
            checkpoint = experiment / "models" / family / f"seed-{seed}.pt"
            predictor = load_checkpoint(checkpoint, manifest)
            manager_class = MLPManager if family == "mlp" else LSTMManager
            candidates[family].append(manager_class(predictor, catalogue, config))
            settings[family].append({"seed": seed, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest()})

    selected, trials = select_managers(candidates, scenarios, policy, config, workers=args.workers)
    winners = {}

    for row in trials:
        family, index = row["family"], row["candidate"]
        row["settings"] = settings[family][index]

        if selected[family] is candidates[family][index]:
            winners[family] = {key: value for key, value in row.items() if key != "days"}

    for family in ("fixed", "adaptive"):
        rows = [r for r in cached_trials if r["family"] == family]

        if rows:
            feasible = [r for r in rows if r["mean_period_abandonment"] <= config.evaluation["abandonment_limit"]]
            winner = min(feasible, key=lambda r: (-r["mean_daily_profit"], r["candidate"])) if feasible else min(
                rows, key=lambda r: (r["mean_period_abandonment"], -r["mean_daily_profit"], r["candidate"]))
            winners[family] = {key: value for key, value in winner.items() if key != "days"}

    output.mkdir(parents=True)
    _save_report({"manifest": asdict(manifest), "selected": winners, "trials": cached_trials + trials,
                  "baseline_selection_sha256": sha256(args.baseline_selection.read_bytes()).hexdigest()
                  if args.baseline_selection else None}, output / "selection.json")
    print(json.dumps(winners, indent=2))


def _evaluate_managers(args: argparse.Namespace) -> None:
    """Freeze validation choices, then report all requested seeds on untouched test days."""
    from .data.storage import load_arrays
    from .evaluate import evaluate, paired_profit_intervals, measure_decision_latency, _write_csv
    from .managers.adaptive import AdaptiveManager
    from .managers.mlp import MLPManager
    from .managers.lstm import LSTMManager
    from .models.training import load_checkpoint
    from .models.evaluation import evaluate_forecasts
    import torch

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    config, manifest, catalogue, policy, scenarios = _management_inputs(args.experiment_dir, args.customer_dir, "test")

    if selection["manifest"] != json.loads(json.dumps(asdict(manifest))):
        raise ValueError("selection belongs to another dataset")

    winners = selection["selected"]
    managers = {"fixed": FixedManager(catalogue[winners["fixed"]["settings"]["action_id"]]),
                "adaptive": AdaptiveManager(catalogue, config, **winners["adaptive"]["settings"])}
    forecast_rows, metrics = [], []
    checkpoints = {}

    for family, manager_class in (("mlp", MLPManager), ("lstm", LSTMManager)):
        trials = sorted((r for r in selection["trials"] if r["family"] == family), key=lambda r: r["settings"]["seed"])

        if len(trials) != 3 or {r["settings"]["seed"] for r in trials} != {0, 1, 2}:
            raise ValueError(f"final evaluation requires validation results for {family} seeds 0, 1 and 2")

        for trial in trials:
            seed = trial["settings"]["seed"]
            checkpoint = args.experiment_dir / "models" / family / f"seed-{seed}.pt"
            digest = sha256(checkpoint.read_bytes()).hexdigest()

            if digest != trial["settings"]["checkpoint_sha256"]:
                raise ValueError(f"{checkpoint}: checkpoint changed after validation")

            label = f"{family}-seed-{seed}"
            predictor = load_checkpoint(checkpoint, manifest)
            managers[label] = manager_class(predictor, catalogue, config)
            checkpoints[label] = digest

    # No selection or tuning occurs below this point; all six checkpoints are retained.
    for split in ("validation", "test"):
        arrays = load_arrays(args.experiment_dir / "dataset" / f"{split}.npz",
                             H=config.model["history_length"], F=config.model["feature_count"])
        splits = json.loads((args.experiment_dir / "dataset" / "splits.json").read_text())
        periods = config.simulation["day_minutes"] // config.simulation["period_minutes"]
        modes = ("uniform", "hold")[:config.dataset["trajectories_per_day"]]
        expected = {(s, mode, i) for s in splits if splits[s] == split for mode in modes for i in range(periods)}

        if set(arrays.keys.tolist()) != expected:
            raise ValueError(f"{split}: forecast keys do not match scenario split")

        for label, manager in managers.items():
            if not hasattr(manager, "predictor"):
                continue

            rows, scores = evaluate_forecasts(manager.predictor, arrays)
            forecast_rows.extend({"split": split, "manager": label, **r} for r in rows)
            metrics.append({"split": split, "manager": label, **scores})

    manifest = replace(manifest, scenario_ids=tuple(s.scenario_id for s in scenarios),
                       parent_hashes={**manifest.parent_hashes, **checkpoints,
                                      "selection": sha256(selection_bytes).hexdigest(),
                                      "test.npz": sha256((args.experiment_dir / "dataset" / "test.npz").read_bytes()).hexdigest()},
                       package_versions={"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__})
    days = evaluate(managers, scenarios, policy, config, output_dir=args.output_dir, manifest=manifest)
    intervals = []

    for baseline in ("fixed", "adaptive"):
        for mode in ("mean_per_day", "separate"):
            intervals.extend(paired_profit_intervals(days, baseline, seed=config.seeds["split"], seed_mode=mode))

    _write_csv(forecast_rows, args.output_dir / "forecasts.csv")
    _write_csv(metrics, args.output_dir / "forecast_metrics.csv")
    _write_csv(intervals, args.output_dir / "paired_profit.csv")

    import copy
    import csv

    # Measure a separate manager on one chronological validation trajectory.
    validation = load_arrays(args.experiment_dir / "dataset" / "validation.npz",
                             H=config.model["history_length"], F=config.model["feature_count"])
    first = validation.keys[0]
    indices = np.flatnonzero((validation.keys["scenario_id"] == first["scenario_id"])
                             & (validation.keys["trajectory_id"] == first["trajectory_id"]))
    indices = indices[np.argsort(validation.keys["period_index"][indices])]
    timing_path = args.experiment_dir / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {"dataset_generation_seconds": None}
    complexity = []

    for label, manager in managers.items():
        row = {**timing, "manager": label, "device": "cpu", "threads": torch.get_num_threads(),
               **measure_decision_latency(copy.deepcopy(manager), validation.history[indices])}
        predictor = getattr(manager, "predictor", None)
        row["parameters"] = sum(p.numel() for p in predictor.parameters()) if predictor is not None else 0

        if predictor is not None:
            family, seed = label.split("-seed-")

            with (args.experiment_dir / "models" / family / f"seed-{seed}-epochs.csv").open() as stream:
                epochs = list(csv.DictReader(stream))

            row.update(training_seconds=float(epochs[-1]["elapsed_seconds"]), epochs=len(epochs),
                       best_epoch=int(epochs[-1]["best_epoch"]))

        complexity.append(row)

    _write_csv(complexity, args.output_dir / "complexity.csv")
    _save_report({"device": "cpu", "seed_policy": "all seeds; paired intervals average seeds within each day, plus separate seed intervals",
                  "selected_on_validation": winners, "independent_test_days": len(scenarios)}, args.output_dir / "protocol.json")
    print(f"Saved matched test results to {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch experiment stages using explicit configuration and artifact paths."""
    parser = argparse.ArgumentParser(prog="python -m cafe.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--config", type=Path, default=Path("config/smoke.json"))
    command.add_argument("--output-dir", type=Path, default=Path("runs/smoke"))
    command.add_argument("--rollout", action="store_true", help="also check the real simulator")

    train = sub.add_parser("train-customers", help="train and select frozen customer Q-tables")
    train.add_argument("--config", type=Path, default=Path("config/default.json"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/customer-training"))

    data = sub.add_parser("generate-data", help="collect and prepare factual management periods")
    data.add_argument("--config", type=Path, default=Path("config/default.json"))
    data.add_argument("--customer-dir", type=Path, required=True, help="selected customer training experiment")
    data.add_argument("--output-dir", type=Path, required=True, help="new dataset experiment directory")

    model = sub.add_parser("train", help="fit a supervised outcome predictor")
    model.add_argument("--experiment-dir", type=Path, required=True)
    model.add_argument("--model", choices=("mlp", "lstm"), default="mlp")
    model.add_argument("--device", default="cpu")
    model.add_argument("--seed", type=int, default=0)
    validation = sub.add_parser("validate-managers", help="rank managers on matched validation days")
    validation.add_argument("--experiment-dir", type=Path, required=True)
    validation.add_argument("--customer-dir", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--seed", type=int, default=0)
    validation.add_argument("--workers", type=int, default=1)
    validation.add_argument("--seeds", type=int, nargs="+")
    validation.add_argument("--models", choices=("mlp", "lstm"), nargs="+", default=["mlp"])
    validation.add_argument("--baseline-selection", type=Path, help="reuse checked fixed/adaptive validation trials")
    evaluation = sub.add_parser("evaluate", help="report matched held-out days for all three neural seeds")
    evaluation.add_argument("--experiment-dir", type=Path, required=True)
    evaluation.add_argument("--customer-dir", type=Path, required=True)
    evaluation.add_argument("--selection", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    report = sub.add_parser("report", help="build static figures and uncertainty tables from saved results")
    report.add_argument("--results-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    replay = sub.add_parser("replay", help="render a synchronized PNG or MP4")
    source = replay.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--input", type=Path)
    replay.add_argument("--at")
    replay.add_argument("--start")
    replay.add_argument("--end")
    replay.add_argument("--speed", type=float, default=2)
    replay.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "replay":
        from .replay import render

        render(args)

        return 0


    if args.command == "report":
        from .report import build_report

        paths = build_report(args.results_dir, args.output_dir)
        print(f"Saved {len(paths)} report artifacts to {args.output_dir}")

        return 0

    if args.command == "evaluate":
        _evaluate_managers(args)

        return 0

    if args.command == "validate-managers":
        _validate_managers(args)

        return 0

    if args.command == "train":
        _train_model(args)

        return 0

    config = load_config(args.config)

    if args.command == "generate-data":
        _generate_data(config, args.customer_dir, args.output_dir)

        return 0

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
