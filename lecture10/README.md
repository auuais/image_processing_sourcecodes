# Lecture 10

This folder contains the lecture 10 practice code for the calibration and lens-distortion implementation slides at the end of `lecture10.pdf`.

## Files

- `common.py`: shared helpers, official dataset download, and calibration routines
- `page110_step_by_step_demonstration.py`: reproduces the demonstration flow from pages 110-114
- `page116_pinhole_camera_model_calibration.py`: monocular pinhole calibration
- `page117_fisheye_camera_model_calibration.py`: fisheye calibration
- `page118_stereo_rig_calibration.py`: stereo rig calibration
- `page119_distorting_and_undistorting_points.py`: point undistortion and reprojection
- `page120_removing_lens_distortion_effects.py`: image undistortion

## Local data

The scripts download their own local resources into `data/` when first run.

- `data/pinhole_calib`: OpenCV sample chessboard images, saved locally as `img_00.png` and up
- `data/fisheyes`: OpenCV fisheye calibration images, saved locally as `Fisheye1_00.png` and up
- `data/stereo/case1`: OpenCV stereo calibration pairs, saved locally as `left*.png` and `right*.png`

Sources:

- `opencv/samples/data` for the monocular and stereo chessboard images
- `opencv_extra/testdata/cv/cameracalibration/fisheye/calib-3_stereo_from_JY/left` for the fisheye images

## How to run

Run any script from inside `lecture10`:

```powershell
python page116_pinhole_camera_model_calibration.py
```

Optional flags:

- `--show` opens the Matplotlib figure after saving

## Output

- Figures and generated arrays are saved in `output/` and the relevant `data/` folders
- The calibration matrices are saved as `camera_mat.npy` and `dist_coefs.npy`
- Stereo calibration is saved as `data/stereo/case1/stereo_case1.npz`

## Note

This lecture PDF does not include a separate assignment slide after the implementation section. The final implementation page, page 120, is treated as the end-of-lecture task.
