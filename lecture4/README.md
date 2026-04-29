# Lecture 4 Practice Code

This folder contains runnable practice code based on `lecture4.pdf`.

## Files

- `data/`: local standard sample images used by the practice scripts
- `page64_sobel_filter.py` to `page71_morphological_filter.py`: practice examples matching the code slides
- `page72_assignment.py`: completed solution for the final assignment slide
- `output/`: generated result figures and assignment outputs

## Dataset choices

- `camera.png`: Sobel, Gabor, DFT, and assignment image filtering
- `astronaut.png`: color unsharp-mask example
- `moon.png`: frequency-based low-pass filtering
- `page.png`: thresholding example
- `binary_blobs.png`: morphological filtering example

## Run examples

```powershell
python page64_sobel_filter.py
python page72_assignment.py
python page72_assignment.py --interactive
```

## Output

- Each practice script saves a summary figure into `output/`
- The assignment script saves individual results into `output/page72_assignment/`
- The assignment script also saves two summary figures:
  - `output/page72_assignment_image_filtering.png`
  - `output/page72_assignment_frequency_filtering.png`
