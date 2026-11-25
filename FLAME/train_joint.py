"""Train LAD-Multi jointly on MagicBrush and SID."""

import argparse
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from run_flame import DATASET_PATHS, JOINT_BATCH_SIZE, PATH_PROFILE, SAM_CHECKPOINT, run_script
from utils.early_stopping import add_early_stopping_arguments, early_stopping_cli_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", "--epoch", type=int, default=3,
                        help="Maximum number of training epochs (default: 3)")
    parser.add_argument("--authentic_ratio", "--authentic-ratio", type=float, default=0.0,
                        help="Authentic-image fraction sampled from source/, 0 disables authentic samples (default: 0)")
    parser.add_argument("--lambda_detection", "--lambda-detection", type=float, default=1.0,
                        help="Image-level detection loss weight when authentic samples are enabled (default: 1)")
    add_early_stopping_arguments(parser)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 0.0 <= args.authentic_ratio <= 1.0:
        parser.error("--authentic-ratio must be between 0 and 1")
    if not isfinite(args.lambda_detection) or args.lambda_detection < 0.0:
        parser.error("--lambda-detection must be finite and nonnegative")
    try:
        early_stopping_args = early_stopping_cli_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    train_num_workers, val_num_workers = (4, 2) if PATH_PROFILE == "server" else (0, 0)
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

    if args.authentic_ratio > 0.0:
        source_dirs = [path / "source" for path in train_paths.values()]
        if not any(
            image.is_file() and image.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for source in source_dirs if source.is_dir()
            for image in source.iterdir()
        ):
            parser.error("Authentic training requires images in a training dataset's source/ directory")

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
            batch_size=JOINT_BATCH_SIZE,
            num_workers=train_num_workers,
            shuffle=True,
            drop_last=False,
            persistent_workers=train_num_workers > 0,
        ),
        val_loader=DataLoaderConfig(
            batch_size=JOINT_BATCH_SIZE,
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
            "--authentic_ratio", str(args.authentic_ratio),
            "--lambda_detection", str(args.lambda_detection),
            "--epochs", str(args.epochs),
            "--batch_size", str(JOINT_BATCH_SIZE),
            "--val_steps", "2000",
            *early_stopping_args,
        ])


if __name__ == "__main__":
    raise SystemExit(main())
