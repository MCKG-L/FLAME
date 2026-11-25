"""Optional early stopping based on completed validation checks."""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


TRAINING_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "training_default.json"


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.002
    min_epochs: int = 1
    reference_score: float | None = field(default=None, init=False)
    bad_checks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("early_stopping.patience must be a positive integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be finite and nonnegative")
        if type(self.min_epochs) is not int or self.min_epochs < 0:
            raise ValueError("early_stopping.min_epochs must be a nonnegative integer")

    @property
    def should_stop(self) -> bool:
        return self.bad_checks >= self.patience

    def update(self, score: float, epoch: int) -> bool:
        if not math.isfinite(score):
            raise ValueError("Early stopping requires a finite validation IoU + F1 score")
        if epoch <= self.min_epochs:
            if self.reference_score is None or score > self.reference_score:
                self.reference_score = score
            self.bad_checks = 0
        elif self.reference_score is None or score > self.reference_score + self.min_delta:
            self.reference_score = score
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.should_stop

    def state_dict(self) -> dict:
        return {"reference_score": self.reference_score, "bad_checks": self.bad_checks}

    def load_state_dict(self, state: dict) -> None:
        reference_score = state.get("reference_score")
        bad_checks = state.get("bad_checks", 0)
        if reference_score is not None and not math.isfinite(reference_score):
            raise ValueError("Invalid early-stopping reference score in checkpoint")
        if not isinstance(bad_checks, int) or bad_checks < 0:
            raise ValueError("Invalid early-stopping counter in checkpoint")
        self.reference_score = reference_score
        self.bad_checks = bad_checks


def add_early_stopping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--early-stopping", "--early_stopping", action="store_true",
                        help="Enable IoU + F1 early stopping using configs/training_default.json")


def early_stopping_from_args(args: argparse.Namespace) -> EarlyStopping | None:
    if not args.early_stopping:
        return None
    try:
        config = json.loads(TRAINING_DEFAULT_CONFIG.read_text(encoding="utf-8"))
        settings = config["early_stopping"]
        return EarlyStopping(
            patience=settings["patience"],
            min_delta=settings["min_delta"],
            min_epochs=settings["min_epochs"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Invalid early-stopping configuration in {TRAINING_DEFAULT_CONFIG}: {error}") from error


def early_stopping_cli_arguments(args: argparse.Namespace) -> list[str]:
    if early_stopping_from_args(args) is None:
        return []
    return ["--early-stopping"]
