# Packaging Label Defect Inspection

Midterm project demo for industrial-style packaging and label defect inspection.

## Project idea

This version uses 5 different real package-front images collected from Open Food Facts:

- cookies
- cereal
- chips
- tea
- chocolate

Each label has 4 generated inspection samples:

- 2 PASS samples with pose variation only
- 2 FAIL samples with simulated defects such as missing print, scratches, stains, or corner damage

The pipeline aligns each test image to the correct label reference using SIFT feature matching, then detects defects using:

- image alignment
- absolute difference
- thresholding
- morphology
- connected components

This keeps the project aligned with the course material learned up to week 7 while making the dataset more realistic than the original single synthetic package.

## Folder structure

- `data/reference/`: 5 downloaded reference labels
- `data/samples/`: generated PASS and FAIL test samples
- `data/sample_metadata.csv`: per-sample label, source URL, and expected result
- `source code/generate_example_data.py`: downloads the label images and builds the dataset
- `source code/run_demo.py`: runs the full inspection pipeline and saves all outputs
- `source code/generate_presentation.py`: rebuilds the PowerPoint deck
- `output/reference_gallery.png`: gallery of the 5 reference labels
- `output/summaries/`: per-sample demonstration figures
- `output/matches/`: SIFT correspondence visualizations
- `output/aligned/`: aligned sample images
- `output/masks/`: final binary defect masks
- `output/inspection_results.csv`: numeric summary
- `output/RESULTS.md`: short written report

## Run

```powershell
cd "C:\Users\USER\Documents\course_Translations\Computer_vision\lecture7\midterm_project\packaging_label_defect_inspection"
python "source code\run_demo.py"
python "source code\generate_presentation.py"
```

## Online image source

The package-front images are downloaded from Open Food Facts product image URLs and mapped to their product pages in `data/sample_metadata.csv` and `output/RESULTS.md`.
