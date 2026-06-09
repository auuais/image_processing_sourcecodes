# Frozen-VLM Results

Primary comparable run: `angle`, `synthetic`, `qwen2.5-vl-3b`, `n=96`.

## Zero-perturbation angle results

- `raw`: MAE 49.94 deg, within-5deg accuracy 7.29%
- `pixels`: MAE 2.77 deg, within-5deg accuracy 95.83%
- `text`: MAE 3.02 deg, within-5deg accuracy 90.62%
- `both`: MAE 2.99 deg, within-5deg accuracy 92.71%

Interpretation: the raw text pathway fails badly on the hardened angle scenes, while any explicit classical measurement channel collapses MAE from about 50 deg to about 3 deg without training a probe or LoRA.

## Trust-under-error

- `|delta|=5`: `pixels` acc 49.48% / follow 94.79%; `text` acc 55.21% / follow 77.08%; `both` acc 51.56% / follow 93.23%
- `|delta|=10`: `pixels` acc 1.56% / follow 94.79%; `text` acc 3.65% / follow 80.73%; `both` acc 2.60% / follow 93.23%

Interpretation: the training-free scaffold is effective when faithful, but it is not reliably self-correcting under injected error. In this run, the model follows wrong scaffold values often, and text injection is not worse than pixels on trust.

## Scope

- The strong claim in this repo is the continuous-angle study above.
- `smolvlm2-2.2b` has only a tiny pilot run and is not used for headline claims.
- The measurement-quality appendix remains a precondition, not the dependent variable.
