# Lecture 6 Practice Code

This folder contains runnable practice code based on `lecture6.pdf`.

## Files

- `data/`: local sample images used by the practice scripts
- `page58_kmeans_segmentation.py`: K-means image segmentation
- `page60_watershed_segmentation.py`: watershed segmentation with demo seeds and optional interactive mode
- `page62_grabcut_segmentation.py`: GrabCut segmentation with demo rectangle/scribbles and optional interactive mode
- `page64_canny_edge_detection.py`: Canny edge detection
- `page66_hough_transform.py`: Hough line and circle detection
- `page68_assignment.py`: final assignment solution
- `output/`: generated figures and assignment outputs

## Dataset choices

- `coffee.png`: K-means segmentation practice and assignment comparison
- `astronaut.png`: watershed seeds, Canny, and assignment GrabCut
- `chelsea.png`: practice GrabCut demo
- `line_circle.png`: Hough transform demo

## Run examples

```powershell
python page58_kmeans_segmentation.py
python page60_watershed_segmentation.py --interactive
python page62_grabcut_segmentation.py --interactive
python page68_assignment.py
python page68_assignment.py --interactive
```
