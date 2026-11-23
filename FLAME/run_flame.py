"""Train FLAME, predict a single-image mask, or evaluate a labelled dataset."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


APP_DIR = Path(__file__).resolve().parent
MAGICBRUSH_IMAGES = APP_DIR.parent.parent / "datasets" / "MagicBrush" / "imgs"
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
        "--num_workers", str(args.num_workers),
        "--num_vis_batches", str(args.num_vis_batches),
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
    training.add_argument("--val-dataset", type=Path, default=MAGICBRUSH_IMAGES / "dev")
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--reload-diagnostics-batches", type=int, default=0,
                          help="Compare this many validation batches after saving a best checkpoint")

    inference = modes.add_parser("infer", help="Save a binary mask for one image")
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--output", type=Path, help="Output PNG mask path")
    inference.add_argument("--threshold", type=float, default=0.5)

    testing = modes.add_parser("test", help="Evaluate a dataset with ground-truth masks")
    testing.add_argument("--dataset", type=Path, default=MAGICBRUSH_IMAGES / "test")
    testing.add_argument("--output", type=Path, default=OUTPUT_DIR / "magicbrush_test")
    testing.add_argument("--batch-size", type=int, default=8)
    testing.add_argument("--num-workers", type=int, default=4,
                         help="Number of data-loading processes (default: 4; 0 uses the main process)")
    testing.add_argument("--num-vis-batches", type=int, default=0,
                         help="Number of batches to visualize (default: 0)")
    testing.add_argument("--authentic-ratio", type=float, default=0.0)
    testing.add_argument("--save-all-vis", action="store_true",
                         help="Save visualizations for all samples, overriding --num-vis-batches")

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
conda activate flame
cd /mnt/storage/FLAME

# 复现论文基础模型时，联合使用 MagicBrush train 和 SID sid_train
# 验证使用 MagicBrush dev 和 SID sid_validation。复制整个命令块运行。
conda activate flame
cd /mnt/storage/FLAME

python - <<'PY'
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

app_dir = Path("FLAME").resolve()
sys.path.insert(0, str(app_dir))

from utils.dataset_config import (
    DataLoaderConfig,
    DatasetConfig,
    DatasetManagerConfig,
    save_dataset_config,
)

magicbrush = Path("/mnt/storage/datasets/MagicBrush/imgs")
sid = Path("/mnt/storage/datasets/SID_Set/imgs")
sam_checkpoint = app_dir / "sam2configs" / "sam2.1_hiera_base_plus.pt"

train_batch_size = 4
val_batch_size = 64
val_steps = 2000

train_paths = {
    "magicbrush_multiedit_train": magicbrush / "train",
    "sid_train": sid / "sid_train",
}
val_paths = {
    "magicbrush_multiedit_val": magicbrush / "dev",
    "sid_validation": sid / "sid_validation",
}

if not sam_checkpoint.is_file():
    raise SystemExit(f"缺少 SAM 权重文件 {sam_checkpoint}")

for dataset_path in (*train_paths.values(), *val_paths.values()):
    for folder in ("target", "mask"):
        required = dataset_path / folder
        if not required.is_dir():
            raise SystemExit(f"缺少数据目录 {required}")

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
        num_workers=4,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
    ),
    val_loader=DataLoaderConfig(
        batch_size=val_batch_size,
        num_workers=2,
        shuffle=False,
        drop_last=False,
    ),
)

with TemporaryDirectory(prefix="flame-joint-") as tmp:
    config_path = Path(tmp) / "datasets.json"
    save_dataset_config(config, str(config_path))

    raise SystemExit(subprocess.call([
        sys.executable, str(app_dir / "train.py"),
        "--dataset_config", str(config_path),
        "--sam_backend", "sam2",
        "--sam_config", "sam2.1_hiera_b+.yaml",
        "--sam_ckpt", str(sam_checkpoint),
        "--forensic_operator", "lad_multi",
        "--img_size", "512",
        "--authentic_ratio", "0",
        "--epochs", "3",
        "--batch_size", str(train_batch_size),
        "--val_steps", str(val_steps),
    ]))
PY

# 可选的 MagicBrush 单数据集训练，使用 dev 做验证
python FLAME/run_flame.py train \
  --train-dataset /mnt/storage/datasets/MagicBrush/imgs/train \
  --val-dataset /mnt/storage/datasets/MagicBrush/imgs/dev \
  --epochs 3 \
  --batch-size 1

# MagicBrush 测试
python FLAME/run_flame.py test \
  --dataset /mnt/storage/datasets/MagicBrush/imgs/test \
  --output FLAME/outputs/magicbrush_test \
  --batch-size 8 \
  --num-workers 4 \
  --num-vis-batches 0 \
  --authentic-ratio 0

# 可选的 SID 单数据集训练，使用 sid_validation 做验证
python FLAME/run_flame.py train \
  --train-dataset /mnt/storage/datasets/SID_Set/imgs/sid_train \
  --val-dataset /mnt/storage/datasets/SID_Set/imgs/sid_validation \
  --epochs 3 \
  --batch-size 1

# SID 测试，sid_test 内需有配对的 target/ 和 mask/
python FLAME/run_flame.py test \
  --dataset /mnt/storage/datasets/SID_Set/imgs/sid_test \
  --output FLAME/outputs/sid_test \
  --batch-size 64 \
  --num-workers 4 \
  --num-vis-batches 0 \
  --authentic-ratio 0

# 测试默认用脚本内置的权重路径。要评估本次训练的模型，
# 在对应测试命令末尾追加 --model 和 --config，
# 分别指向该次训练生成的 best_model.pth 和 model_params.json。
# 联合训练后，两条测试命令可以使用同一组权重和配置。

# 对单张图片推理，默认输出到 FLAME/outputs/inference/
python FLAME/run_flame.py infer --image "FLAME/demo_test_set/magicbrush/target/30.png"
"""
