"""Collect factual management trajectories using a frozen customer policy."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..config import Config, config_hash
from ..contracts import Action, Array, ArtifactManifest, PeriodRecord, Scenario, build_catalogue
from ..customers.policy import CustomerPolicy
from ..managers.base import Manager
from ..simulation.scenarios import save_scenario
from ..simulation.simulator import run_day
from .storage import save_records


class ExplorationManager(Manager):
    """Explore uniformly each period or hold a random action for two to four periods."""

    def __init__(self, catalogue: Sequence[Action], seed: int, *, mode: str) -> None:
        """Bind the catalogue and trajectory-specific random stream."""
        if not catalogue or mode not in ("uniform", "hold"):
            raise ValueError("exploration requires a catalogue and mode uniform or hold")

        self.catalogue = tuple(catalogue)
        self.seed = seed
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        """Replay the same action sequence when the day is repeated."""
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self.remaining = 0
        self.action_index = None

    def choose(self, history: Array) -> Action:
        """Sample independently of customers, including their arrival schedule."""
        if self.remaining == 0:
            # A new hold must change the action even when two adjacent draws would tie.
            if self.mode == "hold" and self.action_index is not None and len(self.catalogue) > 1:
                offset = int(self.rng.integers(1, len(self.catalogue)))
                self.action_index = (self.action_index + offset) % len(self.catalogue)
            else:
                self.action_index = int(self.rng.integers(len(self.catalogue)))

            self.remaining = int(self.rng.integers(2, 5)) if self.mode == "hold" else 1

        self.remaining -= 1

        return self.catalogue[self.action_index]


def build_dataset(scenarios: Sequence[Scenario], customer_policy: CustomerPolicy, config: Config,
                  exploration_seeds: Mapping[str, Sequence[int]], output_dir: Path,
                  manifest: ArtifactManifest) -> list[PeriodRecord]:
    """Replay each schedule under independently seeded managers and save the factual rows."""
    ids = [s.scenario_id for s in scenarios]
    count = config.dataset["trajectories_per_day"]
    seeds = [seed for group in exploration_seeds.values() for seed in group]
    fingerprint = customer_policy.fingerprint()

    if not customer_policy.frozen or fingerprint != manifest.parent_hashes.get("customer_policy"):
        raise ValueError("customer_policy must match the frozen policy fingerprint in the manifest")

    if (count not in (1, 2) or len(ids) != config.dataset["scenario_days"]
            or len(set(ids)) != len(ids) or set(ids) != set(exploration_seeds)
            or tuple(ids) != tuple(manifest.scenario_ids) or manifest.config_hash != config_hash(config)
            or len(set(s.seed for s in scenarios)) != len(ids)):
        raise ValueError("scenario IDs, schedules or trajectory count conflict with dataset configuration")

    if len(set(seeds)) != len(seeds) or any(len(group) != count for group in exploration_seeds.values()):
        raise ValueError("each trajectory requires its own exploration seed")

    catalogue = build_catalogue(config)
    records = []

    for scenario in scenarios:
        if scenario.group != "management" or Path(scenario.scenario_id).name != scenario.scenario_id:
            raise ValueError("expected management scenarios with filename-safe IDs")

        save_scenario(scenario, output_dir / "scenarios" / f"{scenario.scenario_id}.npz")

        for mode, seed in zip(("uniform", "hold"), exploration_seeds[scenario.scenario_id]):
            manager = ExplorationManager(catalogue, seed, mode=mode)
            result = run_day(scenario, customer_policy, manager, config, trajectory_id=mode)
            records.extend(result.periods)

    if fingerprint != customer_policy.fingerprint():
        raise ValueError("customer policy changed during collection")

    (output_dir / "dataset").mkdir(exist_ok=True)
    save_records(records, output_dir / "dataset" / "periods.csv")

    return records
