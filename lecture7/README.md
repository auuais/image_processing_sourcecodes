# Lecture 7 Practice Code

This folder contains runnable practice code based on `lecture7.pdf`.

## Files

- `data/scene01.png`: downloaded from OpenCV sample data and used as the main feature-detection image
- `data/scene02_moved.png`: affine-transformed version of `scene01.png`
- `data/scene02_affine.txt`: affine matrix used to generate `scene02_moved.png`
- `page70_harris_fast_corner_detection.py`: Harris and FAST corner detection
- `page71_good_features_to_track.py`: Good Features to Track corner detection
- `page72_draw_keypoints_descriptors_matches.py`: drawing keypoints and matches
- `page73_sift_scale_invariant_keypoints.py`: SIFT keypoint detection across two scenes
- `page74_assignment.py`: assignment solution
- `output/`: generated figures and assignment outputs

## Assignment interpretation

Page 74 asks for:

1. Detect FAST, Harris Corner, Good Features to Track, and SIFT on image 1, then compare their feature locations using different marker shapes.
2. Detect the same features on image 2, which contains motion relative to image 1, then check which detector remains well detected across the two images.

The assignment script uses the known affine transform between `scene01.png` and `scene02_moved.png` to measure repeatability for each detector.

## Run examples

```powershell
python page70_harris_fast_corner_detection.py
python page71_good_features_to_track.py
python page72_draw_keypoints_descriptors_matches.py
python page73_sift_scale_invariant_keypoints.py
python page74_assignment.py
```
