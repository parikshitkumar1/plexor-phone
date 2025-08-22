#!/usr/bin/env python3
"""
Download a YOLO-ready dataset containing only the COCO "cell phone" class using FiftyOne.

- Downloads COCO-2017 train/val splits filtered to images containing "cell phone"
- Exports to YOLOv5/YOLOv8-compatible layout with separate train/val folders
- Writes a minimal data.yaml pointing to the exported images

Usage:
  python scripts/download_coco_phone_fiftyone.py --out datasets/coco_phone_yolo

Then train with Ultralytics:
  pip install ultralytics
  yolo detect train data=datasets/coco_phone_yolo/data.yaml model=yolov8n.pt epochs=50 imgsz=640
"""

import argparse
import os
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.core.labels import Detections


def _first_detections_field(dataset: fo.Dataset) -> str:
    schema = dataset.get_field_schema()
    # Prefer common names, then fallback to first Detections field
    preferred = ["ground_truth", "detections"]
    for name in preferred:
        if name in schema:
            f = schema[name]
            try:
                if issubclass(f.document_type, Detections):
                    return name
            except Exception:
                pass
    for name, f in schema.items():
        try:
            if issubclass(f.document_type, Detections):
                return name
        except Exception:
            continue
    raise RuntimeError("No Detections field found in dataset")


def export_split(ds: fo.Dataset, out_dir: Path, split_name: str):
    out_split = out_dir / split_name
    out_split.mkdir(parents=True, exist_ok=True)
    label_field = _first_detections_field(ds)

    ds.export(
        export_dir=str(out_split),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        # We explicitly export each split to its own root
    )


def write_data_yaml(out_dir: Path):
    data_yaml = out_dir / "data.yaml"
    content = f"""train: {str((out_dir / 'train' / 'images').resolve())}
val: {str((out_dir / 'val' / 'images').resolve())}
nc: 1
names: ["phone"]
"""
    data_yaml.write_text(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/coco_phone_yolo", help="Output directory")
    ap.add_argument("--max_train", type=int, default=None, help="Limit train samples (optional)")
    ap.add_argument("--max_val", type=int, default=None, help="Limit val samples (optional)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    print("Downloading COCO-2017 train filtered to 'cell phone'...")
    trainset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=["cell phone"],
        only_matching=True,
        max_samples=args.max_train,
        persistent=False,
    )
    print(f"Train samples: {len(trainset)}")

    print("Downloading COCO-2017 val filtered to 'cell phone'...")
    valset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["cell phone"],
        only_matching=True,
        max_samples=args.max_val,
        persistent=False,
    )
    print(f"Val samples: {len(valset)}")

    print("Exporting train split to YOLO format...")
    export_split(trainset, out_dir, "train")
    print("Exporting val split to YOLO format...")
    export_split(valset, out_dir, "val")

    write_data_yaml(out_dir)
    print("\nDone. YOLO dataset ready at:", out_dir.resolve())
    print("data.yaml:", (out_dir / "data.yaml").resolve())
    print("Train images:", (out_dir / "train" / "images").resolve())
    print("Val images:", (out_dir / "val" / "images").resolve())


if __name__ == "__main__":
    main()
