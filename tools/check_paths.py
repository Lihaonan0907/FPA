#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check critical paths defined in a config file without running training."""
import argparse
from pathlib import Path

import yaml


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def check_exists(label: str, path: Path, required: bool = True) -> bool:
    ok = path.exists()
    status = "OK" if ok else ("MISSING" if required else "OPTIONAL")
    print(f"[{status}] {label}: {path}")
    return ok or not required


def check_dataset_layout(dataset_root: Path) -> bool:
    ok = True
    for subdir in ("train", "train_new", "train_label_new"):
        ok &= check_exists(f"Dataset subdir `{subdir}`", dataset_root / subdir)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_root, args.config)

    if not config_path.exists():
        print(f"[MISSING] config file: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    yolov5_root = repo_root / "Pytorch3D" / "yolov5"

    required_keys = {
        "dataset_path": "Dataset root",
        "yolo_weights": "YOLOv5 weights",
        "object3d_data_dir": "3D model directory",
        "texture_init_path": "Initial texture image",
        "nps_values_path": "NPS color triplets",
        "data_yaml": "YOLO dataset yaml",
    }
    optional_keys = {
        "texture_checkpoint_path": "Texture checkpoint (created during training if absent)",
        "texture_bg_path": "Texture background image",
        "output_dir": "Output directory",
    }

    ok_all = True
    for key, label in required_keys.items():
        value = cfg.get(key)
        if not value:
            print(f"[MISSING] {label}: config key `{key}` is empty")
            ok_all = False
            continue
        resolved = resolve_path(yolov5_root, value)
        ok_all &= check_exists(label, resolved)
        if key == "dataset_path" and resolved.exists():
            ok_all &= check_dataset_layout(resolved)
        if key == "object3d_data_dir" and resolved.exists():
            ok_all &= check_exists("3D mesh raw.obj", resolved / "raw.obj")

    for key, label in optional_keys.items():
        value = cfg.get(key)
        if value:
            check_exists(label, resolve_path(yolov5_root, value), required=False)

    raise SystemExit(0 if ok_all else 2)


if __name__ == "__main__":
    main()
