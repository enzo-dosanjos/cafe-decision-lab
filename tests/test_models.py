"""Check the MLP contract, masked learning objective, selector and checkpoint reload."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cafe.config import config_hash, load_config
from cafe.contracts import ArtifactManifest, DatasetArrays, ScalerStats, build_catalogue

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise

    torch = None


@unittest.skipIf(torch is None, "install the neural extra to exercise the MLP")
class ModelTests(unittest.TestCase):
    """Use a small causal history and controlled catalogue scores."""

    def setUp(self) -> None:
        """Prepare the documented dimensions and make padding visible after scaling."""
        from cafe.models.mlp import MLPPredictor

        self.config = load_config(Path("config/smoke.json"))
        self.catalogue = build_catalogue(self.config)
        mean = np.ones(30)
        mean[[9, -1]] = 0
        self.stats = ScalerStats(mean, np.ones(30), np.zeros(7), np.ones(7), np.ones(5), ("train",))
        self.model = MLPPredictor(self.stats)
        self.h = torch.zeros(2, 4, 30)
        self.h[:, -1, -1] = 1
        self.a = torch.tensor([[10, 20, 30, 1, 0, 4, 6]] * 2, dtype=torch.float32)

    def test_scaling_padding_bounds_and_gradient(self) -> None:
        """Padding stays zero and sales are bounded by raw stock, including zero stock."""
        captured = []
        hook = self.model.network[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        values = self.model(self.h, self.a)
        hook.remove()
        self.assertTrue((captured[0][:, :90] == 0).all())
        self.assertEqual(float(captured[0][0, 90]), -1)
        self.assertEqual(float(captured[0][0, 119]), 1)
        self.assertTrue((values >= 0).all())
        self.assertTrue((values[:, :3] <= self.a[:, 4:]).all())
        values.sum().backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters()))

    def test_masked_loss_denominators(self) -> None:
        """Average sales, abandonment and valid waits separately, including empty wait batches."""
        from cafe.models.training import outcome_loss

        predicted = torch.tensor([[1., 1., 1., 2., 3.], [1., 1., 1., 2., 100.]])
        targets = torch.zeros_like(predicted)
        mask = torch.ones_like(predicted, dtype=torch.bool)
        mask[1, -1] = False
        self.assertEqual(float(outcome_loss(predicted, targets, mask, torch.ones(5))), 14)
        mask[:, -1] = False
        self.assertEqual(float(outcome_loss(predicted, targets, mask, torch.ones(5))), 5)

    def test_selection_honors_feasibility_and_stable_ties(self) -> None:
        """Tie-break independently of catalogue order and minimize abandonment on fallback."""
        from unittest.mock import patch
        from cafe.managers.selection import select_action

        catalogue = self.catalogue[::-1]
        outcomes = torch.zeros(len(catalogue), 5)
        costs = np.zeros(3)

        with patch.object(self.model, "predict", return_value=outcomes) as predict:
            action, info = select_action(self.model, self.h[0].numpy(), catalogue, costs, 0, 5)
            self.assertEqual(predict.call_count, 1)
            self.assertEqual(action.action_id, 0)
            self.assertTrue(info.predicted_feasible)
            outcomes[:, 3] = 7
            outcomes[1, 3] = 6
            action, info = select_action(self.model, self.h[0].numpy(), catalogue, costs, 0, 5)
            self.assertEqual(action, catalogue[1])
            self.assertTrue(info.fallback_used)
            outcomes[0, 3] = 5
            action, info = select_action(self.model, self.h[0].numpy(), catalogue, costs, 0, 5)
            self.assertEqual(action, catalogue[0])
            self.assertFalse(info.fallback_used)

    def test_checkpoint_round_trip_and_parent_mismatch(self) -> None:
        """Restore the same raw-unit forecasts and reject a different dataset."""
        from cafe.models.training import load_checkpoint, save_checkpoint

        manifest = ArtifactManifest("1", "1", config_hash(self.config), "catalogue", {"train": "abc"},
                                    ("train",), {}, (), (), {})

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(self.model, path, self.config, 1, manifest)
            loaded = load_checkpoint(path, manifest)
            torch.testing.assert_close(loaded.predict(self.h, self.a), self.model.predict(self.h, self.a))

            with self.assertRaisesRegex(ValueError, "lineage"):
                load_checkpoint(path, replace(manifest, parent_hashes={"train": "xyz"}))

    def test_patience_restores_first_checkpoint_and_protects_existing_run(self) -> None:
        """A flat validation curve stops at patience and reloads the earliest minimum."""
        import csv
        from unittest.mock import patch
        from cafe.models.training import train_predictor

        dtype = [("scenario_id", "U10"), ("trajectory_id", "U8"), ("period_index", "i8")]
        keys = np.array([("train", "uniform", i) for i in range(2)], dtype=dtype)
        train = DatasetArrays(self.h.numpy(), self.a.numpy(), np.zeros((2, 5)), np.ones((2, 5), bool), keys)
        valid_keys = keys.copy()
        valid_keys["scenario_id"] = "validation"
        valid = replace(train, keys=valid_keys)
        config = replace(self.config, training={**self.config.training, "max_epochs": 5, "patience": 1})
        manifest = ArtifactManifest("1", "1", config_hash(config), "catalogue", {}, ("train", "validation"),
                                    {}, (), (), {})

        with TemporaryDirectory() as tmp, patch("cafe.models.training._metrics", return_value={"loss": 1.0}):
            output = Path(tmp)
            trained = train_predictor("mlp", train, valid, self.stats, config, seed=0, device="cpu",
                                      output_dir=output, manifest=manifest)

            with (output / "seed-0-epochs.csv").open() as stream:
                epochs = list(csv.DictReader(stream))

            self.assertEqual([r["best_epoch"] for r in epochs], ["1", "1"])
            self.assertFalse(trained.training)

            with self.assertRaises(FileExistsError):
                train_predictor("mlp", train, valid, self.stats, config, seed=0, device="cpu",
                                output_dir=output, manifest=manifest)

    def test_lstm_windows_reset_and_checkpoint_round_trip(self) -> None:
        """Unrelated calls and batch neighbours cannot change a window's prediction."""
        from cafe.models.lstm import LSTMPredictor
        from cafe.models.training import load_checkpoint, save_checkpoint
        from cafe.managers.lstm import LSTMManager

        model = LSTMPredictor(self.stats)
        distinct = torch.randn_like(self.h)
        distinct[:, :, -1] = 1
        batched = model.predict(distinct, self.a)
        singles = torch.cat([model.predict(distinct[i:i + 1], self.a[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(batched, singles)
        torch.testing.assert_close(model.predict(distinct.flip(0), self.a.flip(0)).flip(0), batched)
        reference = model.predict(self.h[:1], self.a[:1]).clone()
        model.predict(torch.randn_like(self.h), self.a)
        torch.testing.assert_close(model.predict(self.h[:1], self.a[:1]), reference)
        torch.testing.assert_close(model.predict(self.h, self.a)[:1], reference)
        values = model(self.h, self.a)
        self.assertTrue((values >= 0).all())
        self.assertTrue((values[:, :3] <= self.a[:, 4:]).all())
        values.sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        manager = LSTMManager(model, self.catalogue, self.config)
        action = manager.choose(self.h[0].numpy())
        manager.reset()
        self.assertIsNone(manager.last_decision)
        self.assertEqual(manager.choose(self.h[0].numpy()), action)
        manifest = ArtifactManifest("1", "1", config_hash(self.config), "catalogue", {}, ("train",), {}, (), (), {})

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lstm.pt"
            save_checkpoint(model, path, self.config, 0, manifest)
            loaded = load_checkpoint(path, manifest)
            self.assertIsInstance(loaded, LSTMPredictor)
            torch.testing.assert_close(loaded.predict(self.h[:1], self.a[:1]), reference)

    def test_forecasts_mask_wait_and_preserve_keys(self) -> None:
        """Invalid waits cannot inflate errors, while factual row identities survive batching."""
        from unittest.mock import patch
        from cafe.models.evaluation import evaluate_forecasts
        from cafe.contracts import TARGET_NAMES

        dtype = [("scenario_id", "U10"), ("trajectory_id", "U8"), ("period_index", "i8")]
        keys = np.array([("test", "uniform", i) for i in range(2)], dtype=dtype)
        targets = np.zeros((2, 5))
        targets[:, 4] = [2, 1000]
        mask = np.ones((2, 5), bool)
        mask[1, 4] = False
        arrays = DatasetArrays(self.h.numpy(), self.a.numpy(), targets, mask, keys)

        with patch.object(self.model, "predict", return_value=torch.zeros(1, 5)):
            rows, metrics = evaluate_forecasts(self.model, arrays, batch_size=1)

        self.assertEqual([row["period_index"] for row in rows], [0, 1])
        self.assertEqual(metrics[f"mae_{TARGET_NAMES[4]}"], 2)
        self.assertEqual(metrics[f"rmse_{TARGET_NAMES[4]}"], 2)
        self.assertIsNone(rows[1][f"observed_{TARGET_NAMES[4]}"])
