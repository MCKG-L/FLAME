"""Print parameter counts for a FLAME checkpoint without running inference."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from test_dataset import load_and_initialize_model, load_model_config


def print_parameter_counts(model: torch.nn.Module) -> None:
    totals = Counter()
    trainable_counts = Counter()
    for name, parameter in model.named_parameters():
        module_name = name.split(".", 1)[0]
        count = parameter.numel()
        totals[module_name] += count
        if parameter.requires_grad:
            trainable_counts[module_name] += count

    total = sum(totals.values())
    trainable = sum(trainable_counts.values())
    print(f"Total parameters:      {total:>12,}")
    print(f"Trainable parameters:  {trainable:>12,}")
    print(f"Frozen parameters:     {total - trainable:>12,}")
    print()
    print(f"{'Module':<28} {'Total':>12} {'Trainable':>12} {'Share':>8}")
    for module_name, count in totals.most_common():
        share = count / total
        print(f"{module_name:<28} {count:>12,} {trainable_counts[module_name]:>12,} {share:>8.2%}")


def main() -> None:
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=checkpoint_dir / "flame_g2_ladmulti_sam2.pth")
    parser.add_argument("--config", type=Path, default=checkpoint_dir / "model_params.json")
    args = parser.parse_args()

    config = load_model_config(str(args.config))
    model = load_and_initialize_model(config, str(args.model), torch.device("cuda"))
    print_parameter_counts(model)


if __name__ == "__main__":
    main()
