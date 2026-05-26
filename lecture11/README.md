# Lecture 11

This folder contains the lecture 11 practice code for the stereo-geometry implementation slides at the end of `lecture11.pdf`.

## Files

- `common.py`: shared helpers, official dataset download, and calibration routines
- `page106_3d_triangulation.py`: synthetic triangulation demo
- `page107_pnp_pose_estimation.py`: relative pose estimation with `solvePnP`
- `page108_stereo_rectification.py`: stereo rectification using calibrated left/right images
- `page109_fundamental_matrix_computation.py`: estimate `F` and derive `E`
- `page110_essential_decomposition.py`: decompose the essential matrix into candidate rotations and translation
- `page111_estimating_disparity_map.py`: compute disparity maps with `StereoBM` and `StereoSGBM`

## Local data

The scripts create their own local resources in `data/` when first run:

- `data/pinhole_calib`: monocular chessboard images and saved calibration arrays
- `data/stereo/case1`: stereo chessboard images and saved stereo calibration arrays
- `data/disparity`: a rectified stereo pair for disparity-map estimation

Sources:

- `opencv/samples/data/left*.jpg` and `right*.jpg` for the monocular/stereo calibration images
- `opencv/samples/data/aloeL.jpg` and `aloeR.jpg` for disparity estimation

## How to run

Run any script from inside `lecture11`:

```powershell
python page111_estimating_disparity_map.py
```

Optional flags:

- `--show` opens the Matplotlib figure after saving

## Output

- Figures are saved in `output/`
- Calibration arrays are saved in `data/pinhole_calib/` and `data/stereo/case1/`

## Note

This lecture PDF does not include a separate assignment slide after the implementation section. The last implementation page, page 111, is treated as the end-of-lecture final task.
