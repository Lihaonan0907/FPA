# Model And Path Settings

All common paths should be changed in `configs/default.yaml`.

| Config key | Meaning | Default |
| --- | --- | --- |
| `device` | Runtime device, for example `cuda:0` or `cpu` | `cuda:0` |
| `dataset_path` | Dataset root containing `train/`, `train_new/`, `train_label_new/` | `datasets/carla_dataset` |
| `data_yaml` | YOLOv5 dataset YAML | `data/attack.yaml` |
| `yolo_weights` | YOLOv5 detector checkpoint | `best.pt` |
| `object3d_data_dir` | 3D model directory containing `raw.obj` | `data/car_model` |
| `texture_init_path` | Initial texture image | `data/3Dmodels/car/ao_di/texture.jpg` |
| `texture_bg_path` | Background texture/reference image for reconstruction/color constraints | `datasets/car/single/highway/train/1.jpg` |
| `texture_checkpoint_path` | Saved diffusion/texture checkpoint | `checkpoints/texture_para.pt` |
| `nps_values_path` | Printable RGB triplet file | `30values.txt` |
| `output_dir` | Generated render/detection output directory | `outputs/gen` |

Relative paths are resolved from `Pytorch3D/yolov5`.

Example:

```yaml
yolo_weights: checkpoints/best.pt
object3d_data_dir: assets/car_model
texture_init_path: assets/textures/texture.jpg
```

If you prefer the top-level placeholder folders, use paths such as:

```yaml
yolo_weights: ../../checkpoints/best.pt
object3d_data_dir: ../../assets/car_model
output_dir: ../../outputs/gen
```
