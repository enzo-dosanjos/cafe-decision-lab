"""Confirm a validation-selected recipe on fresh paired days after the original test was inspected."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from model_comparison import make_model
from cafe.cli import _catalogue_json, _load_dataset_policy, _model_inputs
from cafe.contracts import build_catalogue
from cafe.data.scaling import load_scalers
from cafe.evaluate import select_managers
from cafe.managers.fixed import FixedManager
from cafe.managers.mlp import MLPManager
from cafe.simulation.scenarios import generate_scenario


def main() -> None:
    """Select by mean validation profit under the service constraint and run a fresh comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--customer-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--scenario-prefix", default="confirm-", help="scenario ID prefix used for customer tie-breaking")
    args = parser.parse_args()
    root, study = args.experiment_dir, args.study_dir
    config, manifest, splits = _model_inputs(root)
    catalogue = build_catalogue(config)
    policy, _, used = _load_dataset_policy(
        config, args.customer_dir, _catalogue_json(catalogue)
    )

    if policy.fingerprint() != manifest.parent_hashes["customer_policy"]:
        raise ValueError("selected policy differs from the management dataset")

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
                    config, f"{args.scenario_prefix}{len(days):03}", "management", seed
                )
            )

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
    candidates = {"fixed": [FixedManager(catalogue[4])]}

    for name, rate in dict.fromkeys((winner, ("adam", "0.001"))):
        managers = []
        for seed in range(3):
            model = make_model(name, stats)
            model.load_state_dict(
                torch.load(study / f"{name}-{rate}-{seed}-35.pt", weights_only=True)
            )
            managers.append(MLPManager(model, catalogue, config))

        candidates[f"{name}-{rate}"] = managers

    _, results = select_managers(candidates, days, policy, config)
    (study / "confirmation.json").write_text(
        json.dumps({"winner": winner, "schedule_seed": 60231123, "trials": results})
    )


if __name__ == "__main__":
    main()
