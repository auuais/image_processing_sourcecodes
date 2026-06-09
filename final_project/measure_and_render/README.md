# Measure-and-Render

Measure-and-Render is the lecture-11 final-project implementation for a narrow but defensible research claim:
use classical computer vision to measure task-relevant geometry, render that measurement back into the image, and test whether a frozen VLM benefits more from the pixel channel, the text channel, or both.

The project is positioned against generic scaffolds such as grids, axes, or neural region marks. The differentiator here is that the scaffold is content-adaptive and metric: watershed counts, Hough-style angle measurements, and later ruler or gauge readings.

## Current scope

The repository now contains a reproducible local research suite with two task families:

- `counting`: split touching objects with thresholding, morphology, distance transform, and watershed
- `angle`: detect and fit principal rays with dark-structure segmentation, Hough line proposals, intersection-based vertex estimation, and cluster-level line fitting

For each task, the pipeline produces VLM-ready variants for:

- `raw`
- `pixels_only`
- `text_only`
- `both`
- `grid`

That means the codebase already supports the core RQ2 study design even before a real VLM is attached.

## Research framing

- `RQ1`: does a classical metric scaffold improve performance over a weaker baseline on precise visual tasks?
- `RQ2`: for the same exact measurement, is it better to communicate through rendered pixels, injected text, or both?
- `RQ3`: does a deterministic measure-render-reask loop help further?
- `RQ4`: when the classical CV measurement is wrong or fragile, does the scaffold help or mislead?

The current code directly addresses `RQ1` and prepares the artifacts needed for `RQ2`. The loop study and error-sensitivity analysis are the next publishable extensions.

## Local results

Current suite summary from `output/suite/suite_summary.csv`:

- `counting`: baseline MAE `3.000`, scaffold MAE `0.083`, baseline exact `8.33%`, scaffold exact `91.67%`
- `angle`: baseline MAE `1.372`, scaffold MAE `1.064`, baseline within-1deg `50.00%`, scaffold within-1deg `58.33%`

Interpretation:

- counting already shows a strong scaffold win and is the cleanest local evidence for the project claim
- angle broadens the method beyond counting and shows the same measurement-and-render pattern on geometric reasoning

## Folder structure

- `data/synthetic_counting/`: synthetic counting benchmark plus metadata
- `data/synthetic_angle/`: synthetic angle benchmark plus metadata
- `source code/common.py`: shared paths, image utilities, and metadata readers
- `source code/harness.py`: metric summaries, confidence intervals, and manifest utilities
- `source code/counting_task.py`: counting benchmark and render pipeline
- `source code/angle_task.py`: angle benchmark and render pipeline
- `source code/generate_synthetic_counting_data.py`: counting data generator
- `source code/generate_synthetic_angle_data.py`: angle data generator
- `source code/run_counting_demo.py`: run the counting task only
- `source code/run_angle_demo.py`: run the angle task only
- `source code/run_research_suite.py`: regenerate both tasks and the combined suite outputs
- `source code/vlm_adapters.py`: adapter stub for future VLM evaluation
- `output/counting/`: counting metrics, variants, and diagnostics
- `output/angle/`: angle metrics, variants, and diagnostics
- `output/suite/`: combined manifest, summary CSV, summary markdown, and comparison plot

## Run

From `C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render`:

```powershell
python "source code\run_counting_demo.py"
python "source code\run_angle_demo.py"
python "source code\run_research_suite.py"
```

## What makes this publishable

The project is not "another overlay." The publishable angle is the combination of:

- deterministic classical-CV measurement instead of a generic reference frame
- task-specific rendered metrics instead of generic marks
- a controlled pixel-versus-text channel comparison for the same measurement
- honest failure analysis when classical CV becomes unreliable

That is a realistic workshop-tier contribution if the next evaluation stage is executed carefully.

## Next steps

1. Attach at least one real VLM in `vlm_adapters.py` and run the combined manifest across `raw`, `pixels_only`, `text_only`, `both`, and `grid`.
2. Add the remaining proposal scaffolds: gauge, ruler, and contour-based comparison.
3. Add generic-grid and Set-of-Mark style baselines for a cleaner external comparison table.
4. Run a deterministic loop condition for `RQ3`.
5. Add scaffold-confidence or injected-error studies for `RQ4`.
