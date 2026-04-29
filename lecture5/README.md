# Lecture 5 Practice Code

This folder contains runnable practice code based on `lecture5.pdf`.

## Files

- `data/`: local sample images used by the practice scripts
- `page97_otsu_thresholding.py` to `page104_distance_transform.py`: practice examples matching the code slides
- `page105_assignment.py`: completed solution for the final assignment slide
- `output/`: generated result figures and assignment outputs

## Dataset choices

- `coins.png`: Otsu thresholding example
- `bnw_shapes.png`: contour hierarchy, curves, and point-location examples
- `text.png`: connected components practice example
- `bnw_shapes.png`: contours and connected-components assignment example
- `distance_circles.png`: distance-transform assignment example

## Run examples

```powershell
python page97_otsu_thresholding.py
python page105_assignment.py
python page102_working_with_curves.py --interactive
python page103_point_location.py --interactive
python page105_assignment.py --interactive
```
