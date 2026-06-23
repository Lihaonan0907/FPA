# Dataset Preparation

The training entry expects a dataset root configured by `dataset_path` in `configs/default.yaml`.

Default layout:

```text
datasets/carla_dataset/
├── train/
│   ├── 000001.jpg
│   └── ...
├── train_new/
│   ├── 000001.png
│   └── ...
└── train_label_new/
    ├── 000001.txt
    └── ...
```

Expected meanings:

- `train/`: background or scene images loaded by `get_loader`.
- `train_new/`: mask images used for foreground/background compositing.
- `train_label_new/`: labels aligned with the training images.

The loader logic is preserved in `Pytorch3D/yolov5/main_gen.py`. If your filenames or label format differ, document the conversion here before changing loader code.

Before training, run:

```bash
python tools/check_paths.py --config configs/default.yaml
```

Do not commit CARLA, COCO, VOC, KITTI, BDD100K, Cityscapes, or other third-party datasets unless their license explicitly allows redistribution.
