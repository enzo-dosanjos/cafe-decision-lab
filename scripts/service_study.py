"""Compare service limits, one-period histories and scheduled actions on matched days."""

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import numpy as np
import torch

from confirm_manager import confirmation_days, validation_selection
from model_comparison import make_model, run
from cafe.cli import _catalogue_json, _load_dataset_policy, _model_inputs
from cafe.contracts import Action, Array, build_catalogue
from cafe.data.scaling import load_scalers
from cafe.data.storage import load_arrays
from cafe.evaluate import select_managers
from cafe.managers.adaptive import AdaptiveManager
from cafe.managers.fixed import FixedManager
from cafe.managers.mlp import MLPManager
from cafe.models.training import _tensors
from cafe.simulation.scenarios import load_scenario
from cafe.simulation.simulator import run_day


class ScheduledManager(FixedManager):
    """Use a fixed action in each successive two-hour block."""

    def __init__(self, actions: list[Action]) -> None:
        """Bind the four selected block actions."""
        super().__init__(actions[0])
        self.actions = actions
        self.reset()

    def reset(self) -> None:
        """Restart at opening."""
        super().reset()
        self.period = 0

    def choose(self, history: Array) -> Action:
        """Advance through four half-hour decisions per block."""
        self.action = self.actions[self.period // 4]
        self.period += 1

        return super().choose(history)


class CurrentManager(MLPManager):
    """Expose only the current observation to a single-period predictor."""

    def choose(self, history: Array) -> Action:
        """Score the catalogue with the latest row."""

        return super().choose(history[-1:])


def rank(rows: list[dict], limit: float) -> dict:
    """Choose profit subject to the mean service limit with deterministic fallback."""
    eligible = [r for r in rows if r["mean_period_abandonment"] <= limit]

    if eligible:
        return max(eligible, key=lambda r: r["mean_daily_profit"])

    return min(
        rows, key=lambda r: (r["mean_period_abandonment"], -r["mean_daily_profit"])
    )


def fit_schedule(catalogue, scenarios, policy, config) -> list[Action]:
    """Select one action per two-hour block from complete validation rollouts."""
    # Fit four block choices from constant-action validation rollouts; evaluate the combined schedule afresh.
    if (
        config.simulation["day_minutes"] // config.simulation["period_minutes"]
        != 16
    ):
        raise ValueError(
            "four two-hour blocks require the full 16-period configuration"
        )

    scores = np.zeros((len(catalogue), 4, 2))
    for action in catalogue:
        manager = FixedManager(action)
        for day in scenarios:
            periods = run_day(
                day, policy, manager, config, trajectory_id="block-fit"
            ).periods
            for row in periods:
                scores[action.action_id, row.period_index // 4] += [
                    row.profit,
                    row.abandonment,
                ]

    actions = []
    for block in range(4):
        rows = [
            dict(
                mean_daily_profit=p / len(scenarios),
                mean_period_abandonment=a / (4 * len(scenarios)),
                action=i,
            )
            for i, (p, a) in enumerate(scores[:, block])
        ]
        actions.append(catalogue[rank(rows, 5)["action"]])

    return actions


def main() -> None:
    """Run one independent experiment and save daily paired outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "experiment-dir",
        "customer-dir",
        "study-dir",
        "selection",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)

    parser.add_argument(
        "--mode",
        choices=("2", "5", "10", "profit", "history", "schedule"),
        required=True,
    )
    args = parser.parse_args()
    root = args.experiment_dir
    output = args.output_dir / args.mode
    if output.exists():
        raise FileExistsError(f"Use a fresh experiment directory: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    config, manifest, splits = _model_inputs(root)
    catalogue = build_catalogue(config)
    policy, _, used = _load_dataset_policy(
        config, args.customer_dir, _catalogue_json(catalogue)
    )
    if manifest.parent_hashes["customer_policy"] != policy.fingerprint():
        raise ValueError("selected customers differ from dataset")

    days = confirmation_days(root, config, used)
    val = [
        load_scenario(root / "scenarios" / f"{s}.npz")
        for s in sorted(splits)
        if splits[s] == "validation"
    ]
    limit = {"2": 2, "5": 5, "10": 10, "profit": float("inf")}.get(args.mode, 5)
    settings = replace(
        config, evaluation={**config.evaluation, "abandonment_limit": limit}
    )
    stats = load_scalers(
        root / "dataset/scalers.json", sorted(s for s in splits if splits[s] == "train")
    )
    archived = validation_selection(args.selection, manifest)
    fixed = [r for r in archived["trials"] if r["family"] == "fixed"]
    candidates = {}
    chosen = {}
    output.mkdir()
    if args.mode == "schedule":
        actions = fit_schedule(catalogue, val, policy, config)

        chosen["schedule"] = [a.action_id for a in actions]
        candidates["schedule"] = [ScheduledManager(actions)]
    elif args.mode == "history":
        parts = []
        for split in ("train", "validation"):
            tensors = _tensors(
                load_arrays(root / "dataset" / f"{split}.npz", H=4, F=30), "cpu"
            )
            parts.append((tensors[0][:, -1:], *tensors[1:]))

        for name in ("adam", "rnn"):
            candidates[name] = []
            for seed in range(3):
                run(name, 0.001, seed, *parts, stats, 35, output, history_length=1)
                model = make_model(name, stats, history_length=1)
                model.load_state_dict(
                    torch.load(output / f"{name}-0.001-{seed}-35.pt", weights_only=True)
                )
                candidates[name].append(CurrentManager(model, catalogue, config))
    else:
        chosen["fixed"] = rank(fixed, limit)["settings"]
        candidates["fixed"] = [FixedManager(catalogue[chosen["fixed"]["action_id"]])]
        variants = [
            dict(alpha=a, buffer=b, queue_threshold=q)
            for a, b, q in product((0.3, 0.6), (1.0, 1.25), (3, 6))
        ]
        managers = [AdaptiveManager(catalogue, settings, **v) for v in variants]
        if limit == 5:
            trials = [r for r in archived["trials"] if r["family"] == "adaptive"]
        else:
            _, trials = select_managers({"adaptive": managers}, val, policy, config)

        i = rank(trials, limit)["candidate"]
        chosen["adaptive"] = variants[i]
        candidates["adaptive"] = [managers[i]]
        for name in ("adam", "rnn", "lstm"):
            candidates[name] = []
            for seed in range(3):
                model = make_model(name, stats)
                model.load_state_dict(
                    torch.load(
                        args.study_dir / f"{name}-0.001-{seed}-35.pt", weights_only=True
                    )
                )
                candidates[name].append(MLPManager(model, catalogue, settings))

    # Score every policy at the common reporting threshold, even when its decision limit differs.
    _, results = select_managers(candidates, days, policy, config)
    (output / "results.json").write_text(
        json.dumps({"settings": chosen, "trials": results})
    )


if __name__ == "__main__":
    main()
