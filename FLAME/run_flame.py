"""Train FLAME, predict a single-image mask, or evaluate a labelled dataset."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.early_stopping import add_early_stopping_arguments, early_stopping_cli_arguments

# Select dataset paths and loader defaults for "windows", "server", or "wsl".
PATH_PROFILE = "wsl"

JOINT_BATCH_SIZE = 48 if PATH_PROFILE == "server" else 1
TEST_BATCH_SIZE, TEST_NUM_WORKERS = (64, 2) if PATH_PROFILE == "server" else (1, 0)

APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = {
    "windows": Path(r"E:\Datasets\FLAME"),
    "server": APP_DIR.parent.parent / "datasets",
    "wsl": Path("/mnt/e/Datasets/FLAME"),
}[PATH_PROFILE]

if PATH_PROFILE == "server":
    magicbrush = DATA_ROOT / "MagicBrush" / "imgs"
    sid = DATA_ROOT / "SID_Set" / "imgs"
    DATASET_PATHS = {
        "magicbrush": {"train": magicbrush / "train", "val": magicbrush / "dev", "test": magicbrush / "test"},
        "sid": {"train": sid / "sid_train", "val": sid / "sid_validation", "test": sid / "sid_test"},
    }
else:
    DATASET_PATHS = {
        "magicbrush": {
            "train": DATA_ROOT / "magicbrush_multiedit_train",
            "val": DATA_ROOT / "magicbrush_multiedit_val",
            "test": DATA_ROOT / "magicbrush_multiedit_test",
        },
        "sid": {"train": DATA_ROOT / "sid_train", "val": DATA_ROOT / "sid_validation", "test": DATA_ROOT / "sid_test"},
    }

TEST_DATASET_PATHS = {family: splits["test"] for family, splits in DATASET_PATHS.items()}
TEST_DATASET_PATHS.update({
    "autosplice": DATA_ROOT / "AutoSplice" / "AutoSplice",
    "cocoglide": DATA_ROOT / "CocoGlide",
})

OUTPUT_DIR = APP_DIR / "outputs"
CHECKPOINT = APP_DIR / "checkpoints" / "flame_g2_ladmulti_sam2.pth"
MODEL_CONFIG = APP_DIR / "checkpoints" / "model_params.json"
SAM_CHECKPOINT = APP_DIR / "sam2configs" / "sam2.1_hiera_base_plus.pt"


def run_script(script: str, arguments: list[str]) -> int:
    command = [sys.executable, str(APP_DIR / script), *arguments]
    return subprocess.run(command, cwd=APP_DIR.parent).returncode


def run_training(config, arguments: list[str]) -> int:
    from utils.dataset_config import save_dataset_config

    with TemporaryDirectory(prefix="flame-train-") as temporary_dir:
        config_path = Path(temporary_dir) / "datasets.json"
        save_dataset_config(config, str(config_path))
        return run_script("train.py", [
            "--dataset_config", str(config_path),
            "--sam_backend", "sam2",
            "--sam_config", "sam2.1_hiera_b+.yaml",
            "--sam_ckpt", str(SAM_CHECKPOINT),
            "--forensic_operator", "lad_multi",
            "--img_size", "512",
            *arguments,
        ])


def train(args: argparse.Namespace, early_stopping_args: list[str] | None = None) -> int:
    from utils.dataset_config import DataLoaderConfig, DatasetConfig, DatasetManagerConfig

    config = DatasetManagerConfig(
        train_datasets=[DatasetConfig("train", str(args.train_dataset.resolve()), force_resize=True)],
        val_datasets=[DatasetConfig("val", str(args.val_dataset.resolve()), is_training=False)],
        train_loader=DataLoaderConfig(batch_size=args.batch_size, num_workers=0, drop_last=False),
        val_loader=DataLoaderConfig(batch_size=1, num_workers=0, shuffle=False, drop_last=False),
    )
    image_count = sum(path.is_file() for path in (args.train_dataset / "target").iterdir())
    batches_per_epoch = (image_count + args.batch_size - 1) // args.batch_size
    return run_training(config, [
        "--authentic_ratio", "0",
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--val_steps", str(batches_per_epoch),
        "--reload_diagnostics_batches", str(args.reload_diagnostics_batches),
        *(early_stopping_args or []),
    ])


def infer(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image, to_tensor

    from test_dataset import load_and_initialize_model, load_model_config

    config = load_model_config(str(args.config))
    image_size = int(config.get("training_config", {}).get("img_size", 512))
    device = torch.device("cuda")
    with Image.open(args.image) as original:
        original_size = original.size
        rgb = original.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    inputs = to_tensor(rgb).unsqueeze(0).to(device)
    model = load_and_initialize_model(config, str(args.model), device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type):
        probability = model(inputs).sigmoid()
        mask = (probability[0, 0] > args.threshold).to(torch.uint8).cpu() * 255

    output = args.output or OUTPUT_DIR / "inference" / f"{args.image.stem}_mask.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(mask).resize(original_size, Image.Resampling.NEAREST).save(output)
    print(f"Mask saved to {output.resolve()}")
    return 0


def test(args: argparse.Namespace) -> int:
    command = [
        "--dataset", str(args.dataset.resolve()),
        "--model", str(args.model.resolve()),
        "--config", str(args.config.resolve()),
        "--output", str(args.output.resolve()),
        "--device", "cuda",
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--num_vis_batches", str(args.num_vis_batches),
        "--authentic_ratio", str(args.authentic_ratio),
    ]
    if args.save_all_vis:
        command.append("--save_all_vis")
    return run_script("test_dataset.py", command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Active profile: {PATH_PROFILE}")
    modes = parser.add_subparsers(dest="mode", required=True)

    training = modes.add_parser("train", help="Train LAD-Multi from a prepared dataset")
    training.add_argument("--dataset-family", choices=DATASET_PATHS,
                          help="Use the configured training and validation directories")
    training.add_argument("--train-dataset", type=Path)
    training.add_argument("--val-dataset", type=Path)
    training.add_argument("--batch-size", type=int, default=1, help="Batch size (default: %(default)s)")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--reload-diagnostics-batches", type=int, default=0,
                          help="Compare this many validation batches after saving a best checkpoint")

    joint_training = modes.add_parser("train-joint", help="Train LAD-Multi jointly on MagicBrush and SID")
    joint_training.add_argument("--epochs", "--epoch", type=int, default=3,
                                help="Maximum number of training epochs (default: 3)")
    joint_training.add_argument("--authentic_ratio", "--authentic-ratio", type=float,
                                help="Optional authentic-image fraction, defaults to 0 in train_joint.py")
    joint_training.add_argument("--lambda_detection", "--lambda-detection", type=float,
                                help="Detection loss weight, defaults to 1 in train_joint.py")
    for training_mode in (training, joint_training):
        add_early_stopping_arguments(training_mode)

    inference = modes.add_parser("infer", help="Save a binary mask for one image")
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--output", type=Path, help="Output PNG mask path")
    inference.add_argument("--threshold", type=float, default=0.5)

    testing = modes.add_parser("test", help="Evaluate a dataset with ground-truth masks")
    testing.add_argument("--experiment-dir", type=Path,
                         help="Use best_model.pth and model_params.json from this training experiment")
    testing.add_argument("--dataset-family", choices=TEST_DATASET_PATHS, default="magicbrush",
                         help="Use the configured test directory (default: magicbrush)")
    testing.add_argument("--dataset", type=Path)
    testing.add_argument("--output", type=Path)
    testing.add_argument("--batch-size", type=int, default=TEST_BATCH_SIZE,
                         help="Batch size (default: %(default)s)")
    testing.add_argument("--num-workers", type=int, default=TEST_NUM_WORKERS,
                         help="Number of data-loading processes (default: %(default)s, 0 uses the main process)")
    testing.add_argument("--num-vis-batches", type=int, default=0,
                         help="Number of batches to visualize (default: 0)")
    testing.add_argument("--authentic-ratio", type=float, default=0.0)
    testing.add_argument("--save-all-vis", action="store_true",
                         help="Save visualizations for all samples, overriding --num-vis-batches")

    for mode in (inference, testing):
        mode.add_argument("--model", type=Path)
        mode.add_argument("--config", type=Path)

    args = parser.parse_args()
    early_stopping_args = []
    if args.mode in ("train", "train-joint"):
        try:
            early_stopping_args = early_stopping_cli_arguments(args)
        except ValueError as error:
            parser.error(str(error))
    if args.mode == "train-joint":
        arguments = ["--epochs", str(args.epochs), *early_stopping_args]
        if args.authentic_ratio is not None:
            arguments.extend(["--authentic_ratio", str(args.authentic_ratio)])
        if args.lambda_detection is not None:
            arguments.extend(["--lambda_detection", str(args.lambda_detection)])
        return run_script("train_joint.py", arguments)

    if args.mode == "test" and args.experiment_dir is not None:
        experiment_dir = args.experiment_dir.resolve()
        args.model = args.model or experiment_dir / "best_model.pth"
        args.config = args.config or experiment_dir / "model_params.json"
        args.output = args.output or experiment_dir / "evaluation" / f"{args.dataset_family}_test"
        for path in (args.model, args.config):
            if not path.is_file():
                testing.error(f"File not found: {path}")

    if args.mode in ("infer", "test"):
        args.model = args.model or CHECKPOINT
        args.config = args.config or MODEL_CONFIG

    if args.mode == "infer":
        if not 0.0 <= args.threshold <= 1.0:
            inference.error("--threshold must be between 0 and 1")
        if args.output and (args.output.suffix.lower() != ".png"
                            or args.output.resolve() == args.image.resolve()):
            inference.error("--output must be a PNG path different from the input image")
        return infer(args)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.mode == "train":
        if args.train_dataset is None and args.dataset_family is None:
            training.error("--train-dataset or --dataset-family is required")
        splits = DATASET_PATHS[args.dataset_family or "magicbrush"]
        args.train_dataset = args.train_dataset or splits["train"]
        args.val_dataset = args.val_dataset or splits["val"]
        datasets = (args.train_dataset, args.val_dataset)
    else:
        args.dataset = args.dataset or TEST_DATASET_PATHS[args.dataset_family]
        args.output = args.output or OUTPUT_DIR / f"{args.dataset_family}_test"
        datasets = (args.dataset,)

    for root in datasets:
        if not (root / "target").is_dir() or not (root / "mask").is_dir():
            parser.error(f"Expected target/ and mask/ directories inside {root}")
    if args.mode == "train":
        if args.reload_diagnostics_batches < 0:
            training.error("--reload-diagnostics-batches must be nonnegative")
        if args.epochs < 1 or not any(path.is_file() for path in (args.train_dataset / "target").iterdir()):
            training.error("--epochs must be positive and the training dataset must contain images")
        return train(args, early_stopping_args)
    if args.num_workers < 0:
        testing.error("--num-workers must be nonnegative")
    if args.num_vis_batches < 0:
        testing.error("--num-vis-batches must be nonnegative")
    if not 0.0 <= args.authentic_ratio <= 1.0:
        testing.error("--authentic-ratio must be between 0 and 1")
    return test(args)


if __name__ == "__main__":
    raise SystemExit(main())

r"""
# 在仓库根目录运行，先设置文件顶部的 PATH_PROFILE。
conda activate flame
======================训练===================================
# 联合使用 MagicBrush 和 SID 训练，并分别使用各自的验证集
python FLAME/run_flame.py train-joint

# 加入真图训练
python FLAME/run_flame.py train-joint --authentic_ratio 0.2 --lambda_detection 1.0

# 启用早停，具体参数在 FLAME/configs/training_default.json 中设置。
python FLAME/run_flame.py train-joint --early-stopping --epochs 20

# 加入真图并启用早停
python FLAME/run_flame.py train-joint --authentic_ratio 0.2 --lambda_detection 1.0 --early-stopping

# MagicBrush 单数据集训练，使用配置中的验证集
python FLAME/run_flame.py train --dataset-family magicbrush --epochs 3 --batch-size 1

# SID 单数据集训练，使用 sid_validation 做验证
python FLAME/run_flame.py train --dataset-family sid --epochs 3 --batch-size 1

# 单数据集训练+早停
python FLAME/run_flame.py train --dataset-family magicbrush --epochs 3 --batch-size 1 --early-stopping
=====================================================================
====================使用官网权重测试===================================

# 测试参数--dataset-family: ===> magicbrush sid autosplice cocoglide
python FLAME/run_flame.py test --dataset-family autosplice

==================--authentic-ratio 1.0加入真图测试====================
# 为了评估整图层面的检测ACC和AP需要加入真图作测试
# 结果保存到 FLAME/outputs/magicbrush_test/evaluation_results_authentic-ratio_1.json
python FLAME/run_flame.py test --dataset-family magicbrush --authentic-ratio 1.0

# 不同比例分别保存，例如 evaluation_results_authentic_ratio_0.2.json
python FLAME/run_flame.py test --dataset-family magicbrush --authentic-ratio 0.2


=======================================================================
=====================使用训练得到的权重测试==============================
# 测试 results 中某次训练的 best_model.pth，自动读取同目录的 model_params.json。
# 将 --experiment-dir 改为要评估的实验目录，结果保存在该目录的 evaluation/ 下。

# 使用同一训练权重测试 SID 参数选择 magicbrush sid autosplice cocoglide
python FLAME/run_flame.py test --dataset-family magicbrush \
  --experiment-dir "FLAME/results/20261123_215019_DET_focal-gamma1p0_focal-alpha0p55_prompt-dim64_lr0p001_dropout-rate0p2_lad-tau0p004_forensic-o_8da0f231"

# 加入真图后，JSON 保存在该实验的 evaluation/magicbrush_test/ 下，文件名包含 authentic_ratio
python FLAME/run_flame.py test --dataset-family magicbrush --authentic-ratio 1.0 \
  --experiment-dir "FLAME/results/20261123_215019_DET_focal-gamma1p0_focal-alpha0p55_prompt-dim64_lr0p001_dropout-rate0p2_lad-tau0p004_forensic-o_8da0f231"

# 对单张图片推理，默认输出到 FLAME/outputs/inference/
python FLAME/run_flame.py infer --image "FLAME/demo_test_set/magicbrush/target/30.png"
"""
