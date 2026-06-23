# Blender Mask Tutorial

This file records the expected documentation for preparing masks used by the current training loader. The exact Blender scene settings should be filled in from the original experiment notes.

## Suggested Workflow

1. Import the vehicle mesh used by the renderer.
2. Align the camera and object transform with the training scene convention.
3. Assign a simple material to the target vehicle region.
4. Render the vehicle mask with a solid foreground and transparent or black background.
5. Export masks into the dataset `train_new/` directory.
6. Ensure mask filenames align with `train/` images and `train_label_new/` annotations.

## Notes To Confirm

- Blender version.
- Coordinate convention and scale.
- Camera focal length/FOV.
- Vehicle pose export format.
- Whether mask edges require dilation, erosion, or manual cleanup.
- Whether the same mask is used for all views or generated per camera pose.

Keep screenshots and source `.blend` files out of git unless their license allows redistribution.
