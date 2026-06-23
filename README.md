# Flexible Physical Camouflage Generation Based on a Differential Approach

This repository is a minimally reorganized release version of the research code for **Flexible Physical Camouflage Generation Based on a Differential Approach**.

The project generates physical adversarial camouflage textures for vehicles. The current code path combines a YOLOv5 detector, a PyTorch3D differentiable renderer, a diffusion/generative texture module, and adversarial, total variation, and non-printability losses.

## Important Principle

This release keeps the original algorithm, model structure, training loop, rendering flow, and loss formulas intact. The cleanup focuses on:

- Moving hard-coded paths into YAML configuration.
- Adding GitHub-ready documentation and placeholders.
- Excluding datasets, checkpoints, 3D assets, and generated results from git.
- Keeping the original main entry at `Pytorch3D/yolov5/main_gen.py`.

The original working directory is not modified by this release copy.

## Repository Structure

```text
Flexible-Physical-Camouflage/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/
│   └── default.yaml
├── docs/
│   ├── blender_mask_tutorial.md
│   ├── dataset_preparation.md
│   ├── method_mapping.md
│   └── model_path_setting.md
├── scripts/
│   └── run_train.sh
├── tools/
│   └── check_paths.py
├── data/
│   └── README.md
├── checkpoints/
│   └── README.md
├── assets/
│   └── README.md
├── outputs/
│   └── README.md
└── Pytorch3D/
    └── yolov5/
        ├── main_gen.py
        ├── loss_fca.py
        ├── style_VAE.py
        ├── Object3D.py
        └── data/
```

## Installation

Create an environment that matches the original CUDA, PyTorch, torchvision, and PyTorch3D versions. The local project appears to have used Python 3.9 and CUDA 11.8-era wheels, but you should confirm exact versions before claiming reproduction.

```bash
conda env create -f environment.yml
conda activate fpa
pip install -r requirements.txt
```

### PyTorch3D Note

PyTorch3D is version-sensitive. Install a wheel that matches your Python, CUDA, PyTorch, and torchvision versions exactly. If PyTorch3D import fails, fix the environment first; the renderer cannot run without it.

## External Assets

Do not commit datasets, pretrained weights, large 3D meshes, generated textures, rendered images, videos, or third-party assets without a clear license.

Expected user-supplied assets include:

- YOLOv5 weights, for example `best.pt`.
- Dataset root containing `train/`, `train_new/`, and `train_label_new/`.
- 3D vehicle model directory containing `raw.obj` and texture/material files.
- Initial texture image.
- NPS printable color triplet file, for example `30values.txt`.

See:

- `docs/dataset_preparation.md`
- `docs/model_path_setting.md`
- `docs/blender_mask_tutorial.md`

## Configuration

Edit `configs/default.yaml` before training. Relative paths are resolved from `Pytorch3D/yolov5` to preserve the original layout.

Common fields:

- `device`: `cuda:0` or `cpu`.
- `dataset_path`: dataset root with `train/`, `train_new/`, `train_label_new/`.
- `data_yaml`: YOLO dataset YAML, usually `data/attack.yaml`.
- `yolo_weights`: detector weights.
- `object3d_data_dir`: 3D model directory containing `raw.obj`.
- `texture_init_path`: initial texture image.
- `texture_checkpoint_path`: diffusion/texture checkpoint path.
- `output_dir`: generated render/output directory.
- `batch_size`, `epochs`, `learning_rate`, `diffusion_step_t`.
- `loss_weights`: adversarial, detector, TV, NPS, color, color ratio, and diffusion/fusion weights.
- `renderer`: image size, FOV, pitch/yaw offsets, and TODO placeholders for physical randomization ranges.

Check paths before training:

```bash
python tools/check_paths.py --config configs/default.yaml
```

## Training

```bash
bash scripts/run_train.sh
```

Equivalent command:

```bash
python Pytorch3D/yolov5/main_gen.py --config configs/default.yaml
```

The script keeps the original training flow. It loads the dataset, initializes the texture/diffusion module, renders the textured 3D object, fuses it with the background, forwards through YOLOv5, and optimizes the configured loss combination.

## Main Code Modules

- `Pytorch3D/yolov5/main_gen.py`: main training entry and config loading.
- `Pytorch3D/yolov5/loss_fca.py`: `PatchTransformer`, `ComputeLoss`, `MaxProbExtractor`, `TotalVariationLoss`, `NPSLoss`, and related helpers.
- `Pytorch3D/yolov5/style_VAE.py`: `VAE`, `DiffusionModel`, and reconstruction/diffusion loss helper.
- `Pytorch3D/yolov5/Object3D.py`: PyTorch3D mesh loading, UV texture replacement, camera setup, lighting, material, and rendering.
- `Pytorch3D/yolov5/fusion.py`: rendered object/background compositing helper.
- `Pytorch3D/yolov5/detect.py`: YOLOv5 visualization/inference helper used during training.

For a paper-to-code table, see `docs/method_mapping.md`.

## Outputs

By default, generated files are written under `Pytorch3D/yolov5/outputs/gen` unless you change `output_dir` in the config. The training checkpoint defaults to `Pytorch3D/yolov5/checkpoints/texture_para.pt`.

These outputs are ignored by `.gitignore`.

## Release Notes

- The code still depends on external data and weights; a fresh clone will not run end-to-end until paths are configured.
- The current cleanup does not rewrite the algorithm into a new package. This is intentional to reduce risk.
- Some paper-level evaluation entries, such as AP@0.5, ASR, transferability, and ablation scripts, should be added only after matching the original experimental scripts.
- Renderer randomization fields are documented in config, but any unverified range is marked as TODO rather than silently implemented.

## Citation

```bibtex
@article{li2026flexible,
  title={Flexible Physical Camouflage Generation Based on a Differential Approach},
  author={Li, Yang and Li, Haonan and Tan, Wenyi and Wang, Tingrui and Pan, Quan},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

## Responsible Use

This code is intended for academic research on physical adversarial robustness and defense. Do not use it to evade real-world safety systems or deploy adversarial camouflage outside controlled, authorized research settings.
