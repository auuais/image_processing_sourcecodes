# Lecture 9

This folder contains the lecture 9 practice code for the implementation slides at the end of `lecture9.pdf`.

## Files

- `common.py`: shared helpers and local data generation
- `page111_warping_affine_perspective_transformations.py`: affine, inverse affine, rotation, and perspective warping
- `page112_remapping_arbitrary_transformation.py`: arbitrary remapping with `cv2.remap`
- `page113_tracking_keypoints_between_frames.py`: Lucas-Kanade keypoint tracking on a traffic clip
- `page114_dense_optical_flow_between_two_frames.py`: dense optical flow on the same traffic clip
- `page115_panorama_image_using_many_images.py`: multi-image panorama stitching

## Local data

The scripts generate their own local resources in `data/` when first run:

- `circlesgrid.png`
- `Lena.png`
- `traffic.mp4`
- `panorama/boat1.jpg` to `boat6.jpg`

The panorama inputs now come from the official OpenCV stitching sample set (`boat1` to `boat6`) instead of the earlier synthetic overlap images.

## How to run

Run any script from inside `lecture9`:

```powershell
python page115_panorama_image_using_many_images.py
```

Optional flags:

- `--show` opens the Matplotlib figure after saving
- `page111_warping_affine_perspective_transformations.py` now starts in interactive point-selection mode by default
- `page111_warping_affine_perspective_transformations.py --non-interactive` uses the built-in source points

## Output

- Figures and generated images are saved in `output/`
- The Lucas-Kanade and dense-flow examples also save annotated videos in `output/`

## Note

This lecture PDF does not include a separate assignment slide. The last practical slide, page 115, is implemented as the end-of-lecture final task.
