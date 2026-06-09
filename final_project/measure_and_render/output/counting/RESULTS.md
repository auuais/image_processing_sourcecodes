# Counting Benchmark

Synthetic counting benchmark for Measure-and-Render.

## Aggregate results

- Baseline (`connected_components`) exact accuracy: 8.33%
- Baseline MAE: 3.000 (95% CI: 2.000, 3.917)
- Scaffold (`watershed`) exact accuracy: 91.67%
- Scaffold MAE: 0.083 (95% CI: 0.000, 0.250)

## Per-sample results

| Sample | Ground truth | CC count | Watershed count | Overlap level | Notes |
|---|---:|---:|---:|---|---|
| count_01 | 6 | 6 | 6 | low | touching_pairs=0 |
| count_02 | 7 | 6 | 7 | low | touching_pairs=1 |
| count_03 | 8 | 6 | 9 | medium | touching_pairs=2 |
| count_04 | 9 | 8 | 9 | medium | touching_pairs=2 |
| count_05 | 10 | 7 | 10 | medium | touching_pairs=3 |
| count_06 | 11 | 8 | 11 | medium | touching_pairs=3 |
| count_07 | 8 | 5 | 8 | high | touching_pairs=3 |
| count_08 | 9 | 5 | 9 | high | touching_pairs=4 |
| count_09 | 10 | 6 | 10 | high | touching_pairs=4 |
| count_10 | 12 | 8 | 12 | high | touching_pairs=5 |
| count_11 | 13 | 8 | 13 | very_high | touching_pairs=6 |
| count_12 | 14 | 8 | 14 | very_high | touching_pairs=6 |
