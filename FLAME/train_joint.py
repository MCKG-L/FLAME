"""Train LAD-Multi jointly on MagicBrush and SID."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from run_flame import DATASET_PATHS, PATH_PROFILE, SAM_CHECKPOINT, run_script
from utils.early_stopping import add_early_stopping_arguments, early_stopping_cli_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", "--epoch", type=int, default=3,
                        help="Maximum number of training epochs (default: 3)")
    add_early_stopping_arguments(parser)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    try:
        early_stopping_args = early_stopping_cli_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    train_batch_size, train_num_workers = (4, 4) if PATH_PROFILE == "server" else (1, 0)
    val_batch_size, val_num_workers = (64, 2) if PATH_PROFILE == "server" else (1, 0)
    train_paths = {
        "magicbrush_multiedit_train": DATASET_PATHS["magicbrush"]["train"],
        "sid_train": DATASET_PATHS["sid"]["train"],
    }
    val_paths = {
        "magicbrush_multiedit_val": DATASET_PATHS["magicbrush"]["val"],
        "sid_validation": DATASET_PATHS["sid"]["val"],
    }

    if not SAM_CHECKPOINT.is_file():
        raise SystemExit(f"缺少 SAM 权重文件 {SAM_CHECKPOINT}")

    for dataset_path in (*train_paths.values(), *val_paths.values()):
        for folder in ("target", "mask"):
            required = dataset_path / folder
            if not required.is_dir():
                raise SystemExit(f"缺少数据目录 {required}")

    from utils.dataset_config import DataLoaderConfig, DatasetConfig, DatasetManagerConfig, save_dataset_config

    config = DatasetManagerConfig(
        train_datasets=[
            DatasetConfig(name, str(path), force_resize=True)
            for name, path in train_paths.items()
        ],
        val_datasets=[
            DatasetConfig(name, str(path), is_training=False, force_resize=True)
            for name, path in val_paths.items()
        ],
        train_loader=DataLoaderConfig(
            batch_size=train_batch_size,
            num_workers=train_num_workers,
            shuffle=True,
            drop_last=False,
            persistent_workers=train_num_workers > 0,
        ),
        val_loader=DataLoaderConfig(
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            shuffle=False,
            drop_last=False,
        ),
    )

    with TemporaryDirectory(prefix="flame-joint-") as temporary_dir:
        config_path = Path(temporary_dir) / "datasets.json"
        save_dataset_config(config, str(config_path))
        return run_script("train.py", [
            "--dataset_config", str(config_path),
            "--sam_backend", "sam2",
            "--sam_config", "sam2.1_hiera_b+.yaml",
            "--sam_ckpt", str(SAM_CHECKPOINT),
            "--forensic_operator", "lad_multi",
            "--img_size", "512",
            "--authentic_ratio", "0",
            "--epochs", str(args.epochs),
            "--batch_size", str(train_batch_size),
            "--val_steps", "2000",
            *early_stopping_args,
        ])


if __name__ == "__main__":
    raise SystemExit(main())
