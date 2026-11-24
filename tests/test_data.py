"""Exercise factual dataset collection, split isolation and persisted model inputs."""

from dataclasses import replace
from itertools import groupby
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cafe.cli import main
from cafe.config import load_config
from cafe.contracts import ArtifactManifest, FEATURE_NAMES, Observation, PeriodRecord, build_catalogue
from cafe.data.generate import ExplorationManager
from cafe.data.scaling import fit_scalers, load_scalers
from cafe.data.splits import make_split_manifest, validate_splits
from cafe.data.storage import load_arrays, load_manifest, load_records, save_arrays
from cafe.data.windows import make_windows
from cafe.simulation.scenarios import load_scenario


class ManagementDataTests(unittest.TestCase):
    """Check the isolation rules independently and exercise the normal CLI."""

    def setUp(self) -> None:
        """Build two distinct real observations and their factual actions."""
        self.config = load_config(Path("config/smoke.json"))
        self.catalogue = build_catalogue(self.config)
        self.rows = []

        for index in range(2):
            values = np.zeros(len(FEATURE_NAMES))
            values[0] = index + 1
            values[-1] = 1
            obs = Observation(index, values)
            self.rows.append(PeriodRecord("a", "uniform", index, obs, self.catalogue[index],
                                          np.array([index, 0, 0]), 1, 0, 2 * index, bool(index),
                                          np.zeros(3, dtype=int), 0, 1, 2, -3, {}))

    def test_exploration_replays_and_changes_after_each_hold(self) -> None:
        """Completed hold intervals span two to four periods and reset reproduces them."""
        for mode in ("uniform", "hold"):
            manager = ExplorationManager(self.catalogue, 31, mode=mode)
            history = np.zeros((4, 30))
            actions = [manager.choose(history).action_id for _ in range(100)]
            manager.reset()
            self.assertEqual(actions, [manager.choose(history).action_id for _ in range(100)])

            if mode == "hold":
                lengths = [len(list(group)) for _, group in groupby(actions)]
                self.assertTrue(all(2 <= length <= 4 for length in lengths[:-1]))

    def test_windows_are_causal_and_keyed_by_trajectory(self) -> None:
        """Interleaved records reset at opening and missing or duplicate periods fail."""
        second = [replace(r, trajectory_id="hold") for r in self.rows]
        arrays = make_windows(second[::-1] + self.rows, {"a": "train"}, split="train", H=4)
        self.assertEqual(arrays.keys["period_index"].tolist(), [0, 1, 0, 1])
        self.assertEqual(arrays.target_mask[:, -1].tolist(), [False, True, False, True])
        np.testing.assert_array_equal(arrays.history[:, :, 0], [[0, 0, 0, 1], [0, 0, 1, 2]] * 2)
        np.testing.assert_array_equal(arrays.actions[1], [*self.catalogue[1].prices, 1, *self.catalogue[1].stock])

        for invalid in (self.rows[1:], self.rows + self.rows[:1]):
            with self.assertRaisesRegex(ValueError, "consecutive"):
                make_windows(invalid, {"a": "train"}, split="train", H=4)

        with self.assertRaisesRegex(ValueError, "trajectory lengths"):
            make_windows(self.rows + second[:1], {"a": "train"}, split="train", H=4)

    def test_scaling_counts_real_training_periods_once(self) -> None:
        """Held-out rows and invalid waits cannot affect any fitted statistic."""
        obs = Observation(2, self.rows[0].observation.values)
        rows = [replace(self.rows[0], mean_wait=900), self.rows[1],
                replace(self.rows[1], period_index=2, observation=obs, mean_wait=8),
                replace(self.rows[1], scenario_id="b", mean_wait=1000)]
        stats = fit_scalers(rows, {"a": "train", "b": "test"})
        self.assertAlmostEqual(stats.observation_mean[0], 4 / 3)
        self.assertEqual(stats.target_std[-1], 3)
        self.assertEqual(stats.observation_mean[-1], 0)
        self.assertEqual(stats.observation_std[-1], 1)
        self.assertEqual(stats.training_scenario_ids, ("a",))

        with self.assertRaisesRegex(ValueError, "valid training waits"):
            fit_scalers(rows[:1], {"a": "train"})

    def test_array_loader_rejects_duplicate_keys_and_invalid_masks(self) -> None:
        """Catch row identity and waiting-target errors before neural training."""
        arrays = make_windows(self.rows, {"a": "train"}, split="train", H=4)
        duplicate = arrays.keys.copy()
        duplicate[1] = duplicate[0]
        masks = arrays.target_mask.copy()
        masks[0, -1] = True

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.npz"

            for corrupt, message in ((replace(arrays, keys=duplicate), "unique"),
                                     (replace(arrays, target_mask=masks), "mask")):
                save_arrays(corrupt, path)

                with self.assertRaisesRegex(ValueError, message):
                    load_arrays(path, H=4, F=30)

    def test_splits_are_disjoint_and_order_independent(self) -> None:
        """Use unique scenario identities for both trajectories and reject customer overlap."""
        ids = [str(i) for i in range(20)]
        splits = make_split_manifest(ids, seed=3)
        self.assertEqual(splits, make_split_manifest(ids[::-1], seed=3))
        self.assertEqual([list(splits.values()).count(s) for s in ("train", "validation", "test")], [14, 3, 3])

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits([replace(self.rows[0], scenario_id=i) for i in ids], splits, [ids[0]])

    def test_cli_generates_reloadable_factual_arrays(self) -> None:
        """Round-trip a complete experiment with two exploration modes on the same schedules."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "customers", root / "management"
            main(["train-customers", "--config", "config/smoke.json", "--output-dir", str(source)])
            config = json.loads((source / "config.json").read_text())
            config["dataset"].update(scenario_days=10, trajectories_per_day=2)
            path = root / "management.json"
            path.write_text(json.dumps(config))
            command = ["generate-data", "--config", str(path), "--customer-dir", str(source), "--output-dir", str(output)]
            main(command)
            dataset = output / "dataset"
            timing = json.loads((output / "timing.json").read_text())
            self.assertGreater(timing["dataset_generation_seconds"], 0)
            manifest = ArtifactManifest(**json.loads((output / "manifest.json").read_text()))
            _, splits = load_manifest(dataset, manifest)
            rows = load_records(dataset / "periods.csv")
            self.assertEqual(len(rows), 40)
            self.assertEqual({r.trajectory_id for r in rows}, {"uniform", "hold"})
            source_ids = json.loads((source / "manifest.json").read_text())["scenario_ids"]
            validate_splits(rows, splits, source_ids)
            load_scalers(dataset / "scalers.json", [s for s in splits if splits[s] == "train"])

            for split in set(splits.values()):
                saved = load_arrays(dataset / f"{split}.npz", H=4, F=30)
                expected = make_windows(rows, splits, split=split, H=4)

                for field in ("history", "actions", "targets", "target_mask", "keys"):
                    np.testing.assert_array_equal(getattr(saved, field), getattr(expected, field))

            schedule = load_scenario(output / "scenarios" / "management-00000.npz")
            self.assertEqual(schedule.group, "management")

            with self.assertRaises(FileExistsError):
                main(command)

            with self.assertRaisesRegex(ValueError, "manifest"):
                load_manifest(dataset, replace(manifest, catalogue_hash="wrong"))

            # Even with a compatible catalogue, changed customer preferences invalidate the policy.
            config["customer_types"][0]["waiting_cost"] += 1
            path.write_text(json.dumps(config))
            rejected = root / "rejected"

            with self.assertRaisesRegex(ValueError, "market configuration"):
                main([*command[:-1], str(rejected)])

            self.assertFalse(rejected.exists())
