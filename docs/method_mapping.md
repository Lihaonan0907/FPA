# Paper To Code Mapping

This table is a release-maintenance aid. Items marked TODO should be verified against the final paper and original experiment scripts before publication.

| Paper module | Formula / algorithm | Code file | Function / class | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Diffusion-based texture generation | Diffusion texture optimization | `Pytorch3D/yolov5/style_VAE.py` | `DiffusionModel` | Implemented | Current trainer passes `diffusion_step_t` from config. |
| U-Net denoising backbone | Denoising/generator network | `Pytorch3D/yolov5/style_VAE.py` | `Unet` | Implemented | Architecture preserved. |
| Reconstruction/diffusion auxiliary loss | MSE-style image loss | `Pytorch3D/yolov5/style_VAE.py` | `loss_fn` | Implemented | Formula not changed. |
| UV texture loading and replacement | Texture projection/blending | `Pytorch3D/yolov5/Object3D.py` | `image_render` | Implemented | Texture insertion location kept compatible with original code. |
| Differentiable rendering | Renderer forward pass | `Pytorch3D/yolov5/Object3D.py` | `load_objs_as_meshes`, `MeshRenderer`, `SoftPhongShader` | Implemented | 3D model path moved to config. |
| Camera randomization | Distance/elevation/azimuth from dataset pose | `Pytorch3D/yolov5/Object3D.py` | `image_render` | Implemented | Config exposes pitch/yaw offsets only; ranges need verification. |
| Lighting/material setup | Renderer lighting/material | `Pytorch3D/yolov5/Object3D.py` | `DirectionalLights`, `Materials` | Implemented | Defaults preserved. |
| Image compositing | Rendered object + background/mask | `Pytorch3D/yolov5/main_gen.py` | training loop tensor fusion | Implemented | Core tensor logic preserved. |
| Detector interface | YOLOv5 forward/inference | `Pytorch3D/yolov5/main_gen.py`, `Pytorch3D/yolov5/detect.py` | `attempt_load`, `DetectMultiBackend`, `run` | Implemented | Weights path moved to config. |
| Adversarial loss | YOLO loss/objectness/class confidence | `Pytorch3D/yolov5/loss_fca.py` | `ComputeLoss`, `MaxProbExtractor` | Implemented | Formula preserved. |
| TV smoothness loss | `L_TV` | `Pytorch3D/yolov5/loss_fca.py` | `TotalVariationLoss` | Implemented | Formula preserved. |
| NPS loss | `L_NPS` | `Pytorch3D/yolov5/loss_fca.py` | `NPSLoss` | Implemented | Printable colors file path moved to config. |
| Overall loss | Weighted sum | `Pytorch3D/yolov5/main_gen.py` | `loss = ...` | Implemented | Weights moved to config with original defaults. |
| Training loop | Texture/diffusion optimization | `Pytorch3D/yolov5/main_gen.py` | `attack` | Implemented | Loop structure preserved. |
| AP@0.5 / ASR evaluation | Evaluation metrics | TODO | TODO | TODO | Add only after locating original evaluation scripts. |
| Transferability | Cross-detector evaluation | TODO | TODO | TODO | YOLOv3/FRCNN assets not packaged in this release. |
| Ablation | Loss/module ablations | TODO | TODO | TODO | Config weights support ablations; dedicated scripts still TODO. |
| Visualization | Generated texture/render/detection | `Pytorch3D/yolov5/main_gen.py`, `Pytorch3D/yolov5/detect.py` | `cv_save`, `run` | Implemented | Output path configurable. |
