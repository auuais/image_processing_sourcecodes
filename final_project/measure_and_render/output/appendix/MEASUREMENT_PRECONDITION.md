# Measurement Quality Appendix

These classical-CV numbers are a precondition for the frozen-VLM study, not the dependent variable.

| Task | Baseline | Scaffold | Baseline MAE | Scaffold MAE | Baseline metric | Scaffold metric |
|---|---|---|---:|---:|---:|---:|
| counting | connected_components | watershed_scaffold | 3.354 | 0.062 | 1.04% exact | 93.75% exact |
| angle | naive_hough | vertex_clustered_ray_scaffold | 6.332 | 2.749 | 39.58% within 1 deg | 58.33% within 1 deg |
