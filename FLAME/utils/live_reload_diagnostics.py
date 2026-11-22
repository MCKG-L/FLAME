"""Compare validation outputs before and after reloading a FLAME checkpoint."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class _TensorComparison:
    finite_elements: int = 0
    nonfinite_elements: int = 0
    sum_abs: float = 0.0
    max_abs: float | None = None
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    def update(self, live: torch.Tensor | None, reloaded: torch.Tensor | None, batch: int) -> None:
        live_shape = list(live.shape) if live is not None else None
        reloaded_shape = list(reloaded.shape) if reloaded is not None else None
        if live_shape != reloaded_shape:
            self.mismatches.append({"batch": batch, "live_shape": live_shape, "reload_shape": reloaded_shape})
            return
        if live is None or reloaded is None:
            return

        difference = (live.float() - reloaded.float()).abs()
        finite = difference[torch.isfinite(difference)]
        self.finite_elements += finite.numel()
        self.nonfinite_elements += difference.numel() - finite.numel()
        if finite.numel():
            self.sum_abs += finite.sum(dtype=torch.float64).item()
            maximum = finite.max().item()
            self.max_abs = maximum if self.max_abs is None else max(self.max_abs, maximum)

    def summary(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / self.finite_elements if self.finite_elements else None,
            "finite_elements": self.finite_elements,
            "nonfinite_elements": self.nonfinite_elements,
            "mismatches": self.mismatches,
        }


@torch.no_grad()
def compare_models_on_batches(
    *,
    live_model: nn.Module,
    reload_model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    num_batches: int,
    amp: bool = True,
) -> dict[str, Any]:
    """Compare up to `num_batches` valid batches using the validation forward pass."""
    if num_batches < 1:
        raise ValueError("num_batches must be positive")
    device = torch.device(device)
    amp_enabled = amp and device.type == "cuda"
    comparisons = {name: _TensorComparison() for name in ("logits", "probs", "coarse_mask")}
    batches_compared = 0
    samples_compared = 0
    skipped_batches = 0

    # Preserve mixed training and eval modes, including frozen SAM components.
    modes = {
        module: module.training
        for model in (live_model, reload_model)
        for module in model.modules()
    }
    try:
        live_model.eval()
        reload_model.eval()
        for batch in loader:
            if not batch:
                skipped_batches += 1
                continue
            orig = batch["orig"].to(device, non_blocking=True)
            streams = [stream.to(device, non_blocking=True) for stream in batch.get("streams", ())]
            context = {}
            for name in ("global_image", "norm_coords"):
                value = batch.get(name)
                context[name] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else None

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, cache_enabled=False):
                live_logits, live_extras = live_model(orig, streams, output_extras=True, **context)
                reload_logits, reload_extras = reload_model(orig, streams, output_extras=True, **context)

            comparisons["logits"].update(live_logits, reload_logits, batches_compared)
            comparisons["probs"].update(live_logits.sigmoid(), reload_logits.sigmoid(), batches_compared)
            comparisons["coarse_mask"].update(
                live_extras.get("coarse_mask"), reload_extras.get("coarse_mask"), batches_compared
            )
            samples_compared += orig.shape[0]
            batches_compared += 1
            if batches_compared == num_batches:
                break
    finally:
        for module, training in modes.items():
            module.training = training

    return {
        "batches_compared": batches_compared,
        "samples_compared": samples_compared,
        "skipped_batches": skipped_batches,
        "amp_enabled": amp_enabled,
        "tensors": {name: comparison.summary() for name, comparison in comparisons.items()},
    }
