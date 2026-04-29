# Image Processing Source Codes

This repository contains the lecture practice code, assignments, datasets, and the midterm project produced for an industrial computer vision course.

## Structure

- `lecture3/`: lecture 3 examples and assignment
- `lecture4/`: filtering, Fourier, thresholding, and morphology examples
- `lecture5/`: thresholding, contours, connected components, distance transform, and assignment
- `lecture6/`: segmentation, Canny, Hough transform, and interactive assignment work
- `lecture7/`: feature detection, matching, repeatability assignment, and the midterm project
- `lecture8/`: local feature descriptors, matching, RANSAC, BoW, and traffic-video matching

## Environment

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## Running

Each lecture folder is self-contained. Run scripts from inside the relevant lecture directory so the local `common.py`, `data/`, and `output/` paths resolve correctly.

Example:

```powershell
cd lecture8
python page87_traffic_reference_match_video.py
```

## Notes

- Lecture PDFs are kept locally but excluded from git.
- Generated `output/` folders are excluded from git to keep the repository lean; the scripts can regenerate them.
- `submissions/` contains local upload packages and archives that are kept outside version control.
- Large local-only assets such as `lecture8/data/traffic.mp4` and the midterm `.pptx` are intentionally not tracked.
