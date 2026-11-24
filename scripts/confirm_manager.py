"""Confirm a validation-selected recipe on fresh paired days after the original test was inspected."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from model_comparison import make_model
from cafe.cli import _catalogue_json, _load_dataset_policy, _model_inputs
from cafe.config import Config
from cafe.contracts import ArtifactManifest, Scenario, build_catalogue
from cafe.data.scaling import load_scalers
from cafe.evaluate import select_managers
from cafe.managers.adaptive import AdaptiveManager
from cafe.managers.fixed import FixedManager
from cafe.managers.mlp import MLPManager
from cafe.models.training import load_checkpoint
from cafe.simulation.scenarios import generate_scenario


def validation_selection(path: Path, manifest: ArtifactManifest) -> dict:
    """Require baseline trials and settings from the same management experiment."""
    selection = json.loads(path.read_text())

    if selection["manifest"] != json.loads(json.dumps(asdict(manifest))):
        raise ValueError(f"{path}: baseline selection and dataset lineage differ")

    return selection


def confirmation_days(root: Path, config: Config, used: set[int], prefix: str = "course-confirm-") -> list[Scenario]:
    """Regenerate the published 150 days, excluding training and management schedule seeds."""
    used.update(
        r["seed"]
        for r in json.loads((root / "scenarios/manifest.json").read_text())["scenarios"]
    )
    rng = np.random.default_rng(60231123)
    days = []

    while len(days) < 150:
        seed = int(rng.integers(2**63 - 1))

        if seed not in used:
            used.add(seed)
            days.append(
                generate_scenario(
                    config, f"{prefix}{len(days):03}", "management", seed
                )
            )

    return days


def main() -> None:
    """Select by mean validation profit under the service constraint and run a fresh comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--customer-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--scenario-prefix", default="course-confirm-", help="scenario ID prefix used for customer tie-breaking")
    parser.add_argument("--selection", type=Path, help="override the published validation-selected baseline settings")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root, study = args.experiment_dir, args.study_dir
    config, manifest, splits = _model_inputs(root)
    catalogue = build_catalogue(config)
    policy, _, used = _load_dataset_policy(
        config, args.customer_dir, _catalogue_json(catalogue)
    )

    if policy.fingerprint() != manifest.parent_hashes["customer_policy"]:
        raise ValueError("selected policy differs from the management dataset")

    days = confirmation_days(root, config, used, args.scenario_prefix)

    trials = {}
    for path in sorted(study.glob("business-*.json")):
        name, rate, seed = path.stem.removeprefix("business-").rsplit("-", 2)
        trials.setdefault((name, rate), {})[int(seed)] = json.loads(path.read_text())

    feasible = [
        key
        for key, runs in trials.items()
        if set(runs) == {0, 1, 2}
        and np.mean([r["mean_period_abandonment"] for r in runs.values()])
        <= config.evaluation["abandonment_limit"]
    ]
    winner = max(
        feasible,
        key=lambda key: np.mean([r["mean_daily_profit"] for r in trials[key].values()]),
    )
    stats = load_scalers(
        root / "dataset/scalers.json", sorted(s for s in splits if splits[s] == "train")
    )
    selected = {"fixed": {"settings": {"action_id": 4}},
                "adaptive": {"settings": {"alpha": 0.6, "buffer": 1.0, "queue_threshold": 6}}}

    if args.selection:
        selected = validation_selection(args.selection, manifest)["selected"]

    candidates = {"fixed": [FixedManager(catalogue[selected["fixed"]["settings"]["action_id"]])],
                  "adaptive": [AdaptiveManager(catalogue, config, **selected["adaptive"]["settings"])]}

    for name, rate in dict.fromkeys((winner, ("adam", "0.001"), ("lstm", "0.001"))):
        managers = []
        for seed in range(3):
            if name == "lstm":
                model = load_checkpoint(root / "models" / "lstm" / f"seed-{seed}.pt", manifest)
            else:
                model = make_model(name, stats)
                model.load_state_dict(torch.load(study / f"{name}-{rate}-{seed}-35.pt", weights_only=True))

            managers.append(MLPManager(model, catalogue, config))

        candidates[f"{name}-{rate}"] = managers

    _, results = select_managers(candidates, days, policy, config, workers=args.workers)
    (study / "confirmation.json").write_text(
        json.dumps({"winner": winner, "schedule_seed": 60231123, "trials": results})
    )


if __name__ == "__main__":
    main()
