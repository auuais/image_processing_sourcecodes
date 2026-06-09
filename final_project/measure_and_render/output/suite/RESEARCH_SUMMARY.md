# Measure-and-Render Research Suite

This suite aggregates the current local benchmarks for the project.

## Task summary

| Task | Baseline | Scaffold | Baseline MAE | Scaffold MAE | Baseline metric | Scaffold metric |
|---|---|---|---:|---:|---:|---:|
| counting | connected_components | watershed_scaffold | 3.000 | 0.083 | 8.33% exact | 91.67% exact |
| angle | naive_hough | vertex_clustered_ray_scaffold | 1.372 | 1.064 | 50.00% within 1 deg | 58.33% within 1 deg |

## Research notes

- Counting benchmark already validates the strongest local claim: a content-adaptive classical measurement scaffold can substantially outperform a weaker vision baseline on precise enumeration.
- Angle benchmark broadens the method beyond counting and shows that the same measure-and-render pattern transfers to geometric measurement.
- The suite now emits matched `raw`, `pixels_only`, `text_only`, `both`, and `grid` variants, so the next research step is a real VLM channel study rather than more local benchmark plumbing.
