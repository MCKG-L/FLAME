"""Download SID-Set shards one at a time and export a FLAME training subset."""

import argparse
import re
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


REPO_ID = "saberzl/SID_Set"
TAMPERED_LABEL = 2
BINARY_MASK = [0] + [255] * 255


def export_pair(row: dict, target: Path, mask: Path, staging: Path) -> None:
    staged_target = staging / "target.png"
    staged_mask = staging / "mask.png"
    with Image.open(BytesIO(row["image"]["bytes"])) as image:
        image.convert("RGB").save(staged_target, compress_level=1)
    with Image.open(BytesIO(row["mask"]["bytes"])) as image:
        image.convert("L").point(BINARY_MASK).save(staged_mask)

    # Publish the mask first so an interruption cannot expose a new unpaired target.
    staged_mask.replace(mask)
    staged_target.replace(target)


def export_split(
    split: str,
    limit: int,
    shards: list[str],
    revision: str | None,
    download_dir: Path,
    output_dir: Path,
    completed: set[str],
) -> None:
    target_dir = output_dir / "target"
    mask_dir = output_dir / "mask"
    target_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".sid-export-", dir=output_dir) as temporary_dir:
        staging = Path(temporary_dir)
        for filename in shards:
            path = download_dir / filename
            if path.is_file():
                print(f"{split}: using local shard {path.name}", flush=True)
            else:
                print(f"{split}: downloading {path.name}", flush=True)
                path = Path(hf_hub_download(
                    REPO_ID,
                    filename,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=download_dir,
                    etag_timeout=30,
                    token=False,
                ))

            print(f"{split}: selecting samples from {path.name}", flush=True)
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(
                    batch_size=16, columns=["label", "img_id", "image", "mask"]
                ):
                    matches = batch.filter(pc.equal(batch.column("label"), TAMPERED_LABEL))
                    for row in matches.to_pylist():
                        image_id = row["img_id"]
                        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", image_id):
                            raise ValueError(f"Invalid SID-Set image ID: {image_id!r}")
                        name = f"{image_id}.png"
                        if name in completed:
                            continue
                        if not (row["image"] or {}).get("bytes") or not (row["mask"] or {}).get("bytes"):
                            continue

                        export_pair(row, target_dir / name, mask_dir / name, staging)
                        completed.add(name)
                        count = len(completed)
                        if count == 1 or count % 100 == 0 or count == limit:
                            print(f"{split}: {count:,}/{limit:,} pairs saved", flush=True)
                        if count == limit:
                            print(f"{split}: finished at {output_dir}", flush=True)
                            return

    raise RuntimeError(f"{split}: expected {limit:,} pairs, found {len(completed):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sid_dir = Path(__file__).resolve().parents[2] / "datasets" / "SID_Set"
    parser.add_argument("--download-dir", type=Path, default=sid_dir)
    parser.add_argument("--output-dir", type=Path, default=sid_dir / "imgs")
    parser.add_argument("--train-count", type=int, default=10_000)
    parser.add_argument("--validation-count", type=int, default=528)
    args = parser.parse_args()
    if min(args.train_count, args.validation_count) < 0:
        parser.error("sample counts must be nonnegative")

    info = None
    for split, limit, folder in (
        ("train", args.train_count, "sid_train"),
        ("validation", args.validation_count, "sid_validation"),
    ):
        if limit == 0:
            continue
        output_dir = args.output_dir / folder
        targets = {path.name for path in (output_dir / "target").glob("*.png") if path.is_file()}
        masks = {path.name for path in (output_dir / "mask").glob("*.png") if path.is_file()}
        completed = targets & masks
        print(f"{split}: {len(completed):,}/{limit:,} pairs already available", flush=True)
        if len(completed) >= limit:
            continue
        if info is None:
            print(f"Fetching shard list for {REPO_ID}...", flush=True)
            info = HfApi().dataset_info(REPO_ID, timeout=30, token=False)
        shards = sorted(
            item.rfilename for item in (info.siblings or [])
            if re.fullmatch(rf"data/{split}-\d+-of-\d+\.parquet", item.rfilename)
        )
        if not shards:
            raise RuntimeError(f"No {split} parquet shards found in {REPO_ID}")
        export_split(split, limit, shards, info.sha, args.download_dir, output_dir, completed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted. Rerun the same command to reuse saved pairs and local shards.")
