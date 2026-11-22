"""Train FLAME, predict a single-image mask, or evaluate a labelled dataset."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("E:/Datasets/FLAME")
OUTPUT_DIR = APP_DIR / "outputs"
CHECKPOINT = APP_DIR / "checkpoints" / "flame_g2_ladmulti_sam2.pth"
MODEL_CONFIG = APP_DIR / "checkpoints" / "model_params.json"
SAM_CHECKPOINT = APP_DIR / "sam2configs" / "sam2.1_hiera_base_plus.pt"


def run_script(script: str, arguments: list[str]) -> int:
    command = [sys.executable, str(APP_DIR / script), *arguments]
    return subprocess.run(command, cwd=APP_DIR.parent).returncode


def train(args: argparse.Namespace) -> int:
    from utils.dataset_config import DataLoaderConfig, DatasetConfig, DatasetManagerConfig, save_dataset_config

    config = DatasetManagerConfig(
        train_datasets=[DatasetConfig("train", str(args.train_dataset.resolve()), force_resize=True)],
        val_datasets=[DatasetConfig("val", str(args.val_dataset.resolve()), is_training=False)],
        train_loader=DataLoaderConfig(batch_size=args.batch_size, num_workers=0, drop_last=False),
        val_loader=DataLoaderConfig(batch_size=1, num_workers=0, shuffle=False, drop_last=False),
    )
    image_count = sum(path.is_file() for path in (args.train_dataset / "target").iterdir())
    batches_per_epoch = (image_count + args.batch_size - 1) // args.batch_size
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
            "--authentic_ratio", "0",
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--val_steps", str(batches_per_epoch),
            "--reload_diagnostics_batches", str(args.reload_diagnostics_batches),
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
        "--num_workers", "0",
        "--authentic_ratio", str(args.authentic_ratio),
    ]
    if args.save_all_vis:
        command.append("--save_all_vis")
    return run_script("test_dataset.py", command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    training = modes.add_parser("train", help="Train LAD-Multi from a prepared dataset")
    training.add_argument("--train-dataset", type=Path, required=True)
    training.add_argument("--val-dataset", type=Path, default=DATA_DIR / "magicbrush_multiedit_val")
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--reload-diagnostics-batches", type=int, default=0,
                          help="Compare this many validation batches after saving a best checkpoint")

    inference = modes.add_parser("infer", help="Save a binary mask for one image")
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--output", type=Path, help="Output PNG mask path")
    inference.add_argument("--threshold", type=float, default=0.5)

    testing = modes.add_parser("test", help="Evaluate a dataset with ground-truth masks")
    testing.add_argument("--dataset", type=Path, default=DATA_DIR / "magicbrush_multiedit_test")
    testing.add_argument("--output", type=Path, default=OUTPUT_DIR / "magicbrush_test")
    testing.add_argument("--batch-size", type=int, default=1)
    testing.add_argument("--authentic-ratio", type=float, default=0.0)
    testing.add_argument("--save-all-vis", action="store_true")

    for mode in (inference, testing):
        mode.add_argument("--model", type=Path, default=CHECKPOINT)
        mode.add_argument("--config", type=Path, default=MODEL_CONFIG)

    args = parser.parse_args()
    if args.mode == "infer":
        if not 0.0 <= args.threshold <= 1.0:
            inference.error("--threshold must be between 0 and 1")
        if args.output and (args.output.suffix.lower() != ".png"
                            or args.output.resolve() == args.image.resolve()):
            inference.error("--output must be a PNG path different from the input image")
        return infer(args)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    datasets = (args.train_dataset, args.val_dataset) if args.mode == "train" else (args.dataset,)
    for root in datasets:
        if not (root / "target").is_dir() or not (root / "mask").is_dir():
            parser.error(f"Expected target/ and mask/ directories inside {root}")
    if args.mode == "train":
        if args.reload_diagnostics_batches < 0:
            training.error("--reload-diagnostics-batches must be nonnegative")
        if args.epochs < 1 or not any(path.is_file() for path in (args.train_dataset / "target").iterdir()):
            training.error("--epochs must be positive and the training dataset must contain images")
        return train(args)
    if not 0.0 <= args.authentic_ratio <= 1.0:
        testing.error("--authentic-ratio must be between 0 and 1")
    return test(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""
conda activate flame
cd D:\Develop\project\IMDL\FLAME

# 测试已导出的 MagicBrush test，输出 IoU 和 F1
python FLAME/run_flame.py test

# 对单张图片推理，默认输出到 FLAME/outputs/inference/
python FLAME/run_flame.py infer --image "FLAME/demo_test_set/magicbrush/target/30.png"

# 导出训练集后，用 SAM 2.1 Base+ 初始化并训练 LAD-Multi
python FLAME/run_flame.py train --train-dataset "E:\Datasets\FLAME\magicbrush_multiedit_train" --epochs 3 --batch-size 1

"""

