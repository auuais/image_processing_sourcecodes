# Comparison Protocol

This project is positioned against two primary references:

1. `arXiv:2603.06459` - *Do Foundation Models Know Geometry? Probing Frozen Features for Continuous Physical Measurement*.
2. `arXiv:2510.26865` - *Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench*.

The goal is not to claim the exact same task as either paper. The goal is to align our evaluation variables, parser rules, and comparison table structure so that our training-free scaffold results can be read next to theirs without ambiguity.

## What We Extract From arXiv:2603.06459

The paper studies a pathway-deficit in VLMs: frozen visual features encode continuous geometry that the text pathway fails to express. The abstract reports the headline gap directly:

- frozen linear probe on hand joint angles: `6.1 deg MAE`
- best text output baseline: `20.0 deg MAE`
- LoRA readout: `6.5 deg MAE`

The same abstract and method sections state the broader validation tasks:

- hand joint / finger flexion angles from FreiHAND
- head pose from BIWI
- rigid object pose from YCB-Video
- gaze direction from MPIIFaceGaze
- camera intrinsics and per-bone analyses in appendices

The model zoo explicitly spans fourteen frozen backbones across:

- self-supervised encoders: `DINOv2 ViT-L`, `DINOv3 ViT-L`, `DINOv2 ViT-B`
- contrastive VL encoders: `CLIP ViT-L`, `SigLIP ViT-L`, `SigLIP-B`
- hybrid VL encoders: `SigLIP 2 ViT-L`, `InternViT-300M`
- generative VLMs: `Qwen2.5-VL-3B`, `Qwen2.5-VL-7B`, `QwenVIT-3B`, `QwenVIT-merger`, `Gemma 3 4B-IT`

The core metric is continuous-angle regression quality in degrees, reported primarily as `MAE` and also analyzed with `R^2`.

## Our Alignment To arXiv:2603.06459

We align on the following dimensions:

- dependent variable: `angle MAE in degrees`
- comparison framing: `text pathway` versus `non-text access to geometry`
- model overlap: prioritize `Qwen2.5-VL-3B` and, when hardware allows, `Qwen2.5-VL-7B`
- no-training setting: our method remains `training-free` and `label-free`

We do **not** align on supervision. Their strongest numbers come from trained probes or LoRA. Our claim is narrower:

> A deterministic classical-CV scaffold can bridge part of the same text-pathway deficit without training a new readout head.

Accordingly, our side-by-side table uses these rows:

- their text baseline
- their trained linear probe
- their LoRA readout
- our raw VLM
- our scaffold-pixels
- our scaffold-text
- our scaffold-both

This makes the training-cost difference explicit.

## What We Extract From MeasureBench

MeasureBench is the benchmark reference for practical visual measurement reading.

Key benchmark facts from the paper:

- `2,442` image-question pairs total
- `1,272` real-world and `1,170` synthetic
- `26` real-world instrument types
- readout families: `dial`, `digital`, `linear`, `composite`
- each sample is paired with a reading question

MeasureBench uses interval-based grading instead of strict exact matching:

- parse the final answer from model output
- numeric parsing includes integers, decimals, scientific notation, and fractions
- if multiple scalars appear, use the `rightmost` scalar
- time responses use the first `hh:mm[:ss]` pattern
- correctness is based on whether the value falls inside a ground-truth interval, plus unit matching when required

The benchmark reports `overall accuracy`, `value accuracy`, and `unit accuracy`.

The paper also shows that prompt changes and test-time thinking provide little help, which directly motivates our inclusion of a `cot` baseline but not any expectation that it should rescue fine-grained perception.

## Our Alignment To MeasureBench

We borrow three protocol ideas directly:

- numeric parsing uses the `rightmost scalar`
- trust in textual reasoning alone should not be assumed to help visual measurement
- dial and linear reading remain the long-term extension target for this repo

We do **not** force our current angle and counting tasks into interval grading. Instead:

- counting uses exact accuracy and MAE
- angle uses `MAE in degrees` plus `within 5 deg` accuracy

That is deliberate. The 2603 paper is the closer target for the current angle scaffold, while MeasureBench informs how we parse and later extend to gauges and rulers.

## Models To Run In This Repo

Primary local open-source targets:

- `Qwen2.5-VL-3B-Instruct`
- `SmolVLM2-2.2B-Instruct`

Secondary local targets when dependencies or VRAM allow:

- `Qwen2.5-VL-7B-Instruct`
- `Phi-3.5-vision-instruct`

Rationale:

- `Qwen2.5-VL-3B` overlaps directly with the generative VLM family listed in `2603.06459`
- smaller local models let us run the full condition matrix on a 10 GB GPU
- a second family is necessary to show the effect is not unique to one architecture

## Conditions To Report

For each sample we report:

- `raw`
- `cot`
- `grid`
- `som`
- `pixels`
- `text`
- `both`

For trust-under-error, we additionally report controlled perturbations on:

- `pixels`
- `text`
- `both`

## Main Tables And Figures

The results pipeline should always produce:

1. `output/vlm/summary.csv` - frozen-VLM metrics by condition and model
2. `output/vlm/results_long.csv` - one row per sample x condition x model
3. `output/vlm/*trust_under_error*.png` - accuracy versus injected error and follow-rate
4. `output/comparison/comparison_table.csv` - side-by-side table versus `2603.06459`
5. `output/comparison/COMPARISON.md` - narrative comparison and caveats

## Explicit Non-Claim

The classical measurement appendix is not the result.

`connected-components vs watershed` and `naive-Hough vs ray-clustered scaffold` remain useful only because they tell us whether the classical scaffold is reliable enough to study frozen-VLM behavior. The dependent variable for the paper-quality claim is the frozen-VLM output under each condition.
