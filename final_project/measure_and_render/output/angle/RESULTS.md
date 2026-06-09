# Angle Benchmark

Synthetic angle-measurement benchmark for Measure-and-Render.

## Aggregate results

- Baseline (`naive_hough`) exact accuracy: 0.00%
- Baseline within-1deg accuracy: 50.00%
- Baseline MAE: 1.372 (95% CI: 0.794, 2.045)
- Scaffold (`vertex_clustered_ray_scaffold`) exact accuracy: 0.00%
- Scaffold within-1deg accuracy: 58.33%
- Scaffold MAE: 1.064 (95% CI: 0.593, 1.692)

## Per-sample results

| Sample | Ground truth | Baseline angle | Scaffold angle | Clutter level | Notes |
|---|---:|---:|---:|---|---|
| angle_01 | 24.0 | 23.1 | 23.5 | low | clutter_segments=0; nuisance_shapes=1 |
| angle_02 | 32.0 | 31.9 | 31.9 | low | clutter_segments=1; nuisance_shapes=1 |
| angle_03 | 41.0 | 41.3 | 40.9 | low | clutter_segments=1; nuisance_shapes=2 |
| angle_04 | 55.0 | 55.7 | 55.3 | medium | clutter_segments=2; nuisance_shapes=2 |
| angle_05 | 68.0 | 69.3 | 69.3 | medium | clutter_segments=2; nuisance_shapes=2 |
| angle_06 | 82.0 | 80.2 | 81.3 | medium | clutter_segments=2; nuisance_shapes=3 |
| angle_07 | 97.0 | 96.7 | 96.0 | medium | clutter_segments=3; nuisance_shapes=3 |
| angle_08 | 109.0 | 107.2 | 108.0 | high | clutter_segments=3; nuisance_shapes=3 |
| angle_09 | 121.0 | 118.5 | 120.7 | high | clutter_segments=3; nuisance_shapes=4 |
| angle_10 | 136.0 | 131.9 | 132.5 | high | clutter_segments=4; nuisance_shapes=4 |
| angle_11 | 48.0 | 45.6 | 45.2 | very_high | clutter_segments=4; nuisance_shapes=5 |
| angle_12 | 73.0 | 72.8 | 74.2 | very_high | clutter_segments=5; nuisance_shapes=5 |
