"""Regression checks for optional validation-driven early stopping."""

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import run_flame
import train
import train_joint
import utils.early_stopping as early_stopping_config
from model.ema import EMA
from utils.early_stopping import EarlyStopping, add_early_stopping_arguments, early_stopping_cli_arguments


class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        root = Path(temporary_dir.name)
        (root / "target").mkdir()
        (root / "mask").mkdir()
        checkpoint = root / "sam.pt"
        checkpoint.touch()
        paths = {family: {"train": root, "val": root} for family in ("magicbrush", "sid")}
        for patcher in (
            patch.object(run_flame, "DATASET_PATHS", paths),
            patch.object(train_joint, "DATASET_PATHS", paths),
            patch.object(train_joint, "SAM_CHECKPOINT", checkpoint),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_improvements_accumulate_against_significant_reference(self):
        stopping = EarlyStopping(patience=3, min_delta=0.002, min_epochs=0)
        self.assertFalse(stopping.update(1.0, 1))
        self.assertFalse(stopping.update(1.001, 1))
        self.assertFalse(stopping.update(1.002, 1))
        self.assertEqual(stopping.bad_checks, 2)
        self.assertFalse(stopping.update(1.003, 1))
        self.assertEqual(stopping.bad_checks, 0)

    def test_warmup_and_counter_continue_across_epochs(self):
        stopping = EarlyStopping(patience=2, min_epochs=1)
        for score in (0.8, 0.7, 0.6):
            self.assertFalse(stopping.update(score, 1))
        self.assertEqual(stopping.reference_score, 0.8)
        self.assertFalse(stopping.update(0.79, 2))
        self.assertTrue(stopping.update(0.79, 3))

    def test_state_roundtrip_and_nonfinite_score(self):
        stopping = EarlyStopping(patience=2, min_epochs=0)
        stopping.update(1.0, 1)
        stopping.update(0.9, 1)
        restored = EarlyStopping(patience=2, min_epochs=0)
        restored.load_state_dict(stopping.state_dict())
        self.assertTrue(restored.update(0.9, 2))
        with self.assertRaises(ValueError):
            restored.update(float("nan"), 2)

    def test_checkpoint_includes_optional_state_and_preserves_legacy_payload(self):
        model = torch.nn.Linear(1, 1)
        stopping = EarlyStopping(min_epochs=0)
        stopping.update(1.0, 1)
        stopping.update(0.9, 1)
        legacy = train.build_checkpoint_payload(model=model, epoch=1, batch=1, score=1.0)
        self.assertNotIn("early_stopping", legacy)
        payload = train.build_checkpoint_payload(
            model=model, epoch=1, batch=1, score=1.0,
            early_stopping_state=stopping.state_dict(),
        )
        stopping.update(1.1, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checkpoint.pth"
            torch.save(payload, path)
            restored = torch.load(path, weights_only=True)
        self.assertEqual(restored["early_stopping"], {"reference_score": 1.0, "bad_checks": 1})

    def test_cli_is_disabled_by_default_and_validates_enabled_settings(self):
        parser = argparse.ArgumentParser()
        add_early_stopping_arguments(parser)
        self.assertEqual(early_stopping_cli_arguments(parser.parse_args([])), [])
        args = parser.parse_args(["--early-stopping"])
        self.assertEqual(early_stopping_cli_arguments(args), ["--early-stopping"])
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "training_default.json"
            with patch.object(early_stopping_config, "TRAINING_DEFAULT_CONFIG", path):
                for field, value in (("patience", 0), ("min_delta", float("nan")),
                                     ("min_delta", -1), ("min_epochs", -1)):
                    settings = {"patience": 5, "min_delta": 0.002, "min_epochs": 1, field: value}
                    path.write_text(json.dumps({"early_stopping": settings}))
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        early_stopping_cli_arguments(args)

    def test_enabled_reads_changed_config_and_disabled_does_not_require_it(self):
        parser = argparse.ArgumentParser()
        add_early_stopping_arguments(parser)
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "training_default.json"
            with patch.object(early_stopping_config, "TRAINING_DEFAULT_CONFIG", path):
                self.assertEqual(early_stopping_cli_arguments(parser.parse_args([])), [])
                with self.assertRaisesRegex(ValueError, "training_default.json"):
                    early_stopping_cli_arguments(parser.parse_args(["--early-stopping"]))
                path.write_text(json.dumps({
                    "early_stopping": {"patience": 8, "min_delta": 0.01, "min_epochs": 2},
                }))
                stopping = early_stopping_config.early_stopping_from_args(parser.parse_args(["--early-stopping"]))
                self.assertEqual((stopping.patience, stopping.min_delta, stopping.min_epochs), (8, 0.01, 2))

    def test_run_flame_forwards_optional_arguments_to_both_joint_routes(self):
        with (
            patch("sys.argv", ["run_flame.py", "train-joint"]),
            patch.object(run_flame, "run_script", return_value=0) as launch,
        ):
            self.assertEqual(run_flame.main(), 0)
            launch.assert_called_once_with("train_joint.py", ["--epochs", "3"])
        with (
            patch("sys.argv", ["run_flame.py", "train-joint", "--early-stopping"]),
            patch.object(run_flame, "run_script", return_value=0) as launch,
        ):
            self.assertEqual(run_flame.main(), 0)
            self.assertIn("--early-stopping", launch.call_args.args[1])
        with (
            patch("sys.argv", ["run_flame.py", "train-joint", "--authentic_ratio", "0", "--early-stopping"]),
            patch.object(run_flame, "run_training", return_value=0) as launch,
        ):
            self.assertEqual(run_flame.main(), 0)
            self.assertIn("--early-stopping", launch.call_args.args[1])

    def test_single_dataset_and_standalone_joint_forward_the_flag(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "target").mkdir()
            (root / "mask").mkdir()
            (root / "target" / "image.png").touch()
            command = [
                "run_flame.py", "train", "--train-dataset", str(root),
                "--val-dataset", str(root), "--early-stopping",
            ]
            with patch("sys.argv", command), patch.object(run_flame, "run_training", return_value=0) as launch:
                self.assertEqual(run_flame.main(), 0)
                self.assertIn("--early-stopping", launch.call_args.args[1])
        with (
            patch("sys.argv", ["train_joint.py", "--early-stopping"]),
            patch.object(train_joint, "run_script", return_value=0) as launch,
        ):
            self.assertEqual(train_joint.main(), 0)
            self.assertIn("--early-stopping", launch.call_args.args[1])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, image, streams, output_extras=True):
        logits = self.weight * image[:, :1]
        return logits, {"coarse_mask": logits, "iou_pred": self.weight.sigmoid().expand(len(image), 1)}


class TestTrainingEarlyStopping(unittest.TestCase):
    def run_epoch(self, stopping=None, epoch=1, val_steps=1, scores=None):
        model = TinyModel()
        ema = EMA(model, decay=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        writer = Mock()
        batch = {"orig": torch.ones(1, 3, 8, 8), "mask": torch.ones(1, 1, 8, 8), "streams": []}
        validate = Mock(side_effect=[{"iou": value / 2, "f1": value / 2} for value in (scores or [1.0] * 5)])
        with tempfile.TemporaryDirectory() as temporary_dir, redirect_stderr(io.StringIO()):
            scaler = torch.amp.GradScaler("cpu")
            with patch.object(train.torch.amp, "GradScaler", return_value=scaler):
                result = train.train_epoch(
                    model, [batch] * 5, optimizer, torch.device("cpu"),
                    val_loader=[batch], validate_fn=validate, epoch=epoch,
                    total_epochs=3, val_steps=val_steps,
                    model_save_path=str(Path(temporary_dir) / "best_model.pth"),
                    checkpoint_save_mode="none", ema=ema, writer=writer, early_stopping=stopping,
                )
        self.assertEqual(ema.backup, {})
        self.assertTrue(model.training)
        return result, validate, writer

    def test_disabled_runs_every_batch_with_original_validation_frequency(self):
        result, validate, _ = self.run_epoch(val_steps=2)
        self.assertEqual(len(result[0]), 5)
        self.assertEqual(validate.call_count, 2)

    def test_stops_inside_epoch_after_patience_and_flushes_events(self):
        stopping = EarlyStopping(patience=2, min_epochs=0)
        result, validate, writer = self.run_epoch(stopping)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(validate.call_count, 3)
        self.assertTrue(stopping.should_stop)
        writer.flush.assert_called()

    def test_enabled_validates_at_epoch_end_and_preserves_counter(self):
        stopping = EarlyStopping(patience=1, min_epochs=0)
        self.run_epoch(stopping, epoch=1, val_steps=100)
        self.assertFalse(stopping.should_stop)
        self.run_epoch(stopping, epoch=2, val_steps=100)
        self.assertTrue(stopping.should_stop)

    def test_nonfinite_validation_restores_ema_weights_before_raising(self):
        model = TinyModel()
        ema = EMA(model, decay=0.5)
        batch = {"orig": torch.ones(1, 3, 8, 8), "mask": torch.ones(1, 1, 8, 8), "streams": []}
        validate = Mock(return_value={"iou": float("nan"), "f1": 0.5})
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            redirect_stderr(io.StringIO()),
            self.assertRaises(ValueError),
        ):
            scaler = torch.amp.GradScaler("cpu")
            with patch.object(train.torch.amp, "GradScaler", return_value=scaler):
                train.train_epoch(
                    model, [batch], torch.optim.SGD(model.parameters(), lr=0.1), torch.device("cpu"),
                    val_loader=[batch], validate_fn=validate,
                    model_save_path=str(Path(temporary_dir) / "best_model.pth"),
                    early_stopping=EarlyStopping(min_epochs=0), ema=ema,
                    epoch=1,
                )
        self.assertEqual(ema.backup, {})
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
