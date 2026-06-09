# Measure-and-Render

Measure-and-Render is the lecture-11 final project turned into a frozen-VLM study. The contribution is no longer the classical measurement pipeline by itself. The contribution is a **training-free, label-free bridge** from deterministic classical-CV measurements into the VLM text pathway, evaluated under clean and controlled-error conditions.

## What is implemented

- Classical counting scaffold:
  - saturation thresholding
  - morphology
  - distance transform
  - watershed
  - centroid and moment extraction
- Classical angle scaffold:
  - dark-structure segmentation
  - Hough line proposals
  - vertex estimation from intersections
  - ray-direction clustering
  - `fitLine` refinement
- Frozen-VLM adapters with disk caching:
  - local Hugging Face models
  - Ollama
  - optional OpenAI / Anthropic / Gemini APIs
- De-leaked evaluation conditions:
  - `raw`
  - `cot`
  - `grid`
  - `som`
  - `pixels`
  - `text`
  - `both`
- Trust-under-error perturbations:
  - counting: `delta in {-3, -1, +1, +3}`
  - angle: `delta in {-10, -5, +5, +10}` degrees

## Research framing

The comparable target is [`docs/COMPARISON_PROTOCOL.md`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\docs\COMPARISON_PROTOCOL.md), which aligns this repo to:

- `arXiv:2603.06459` for continuous-angle `MAE` framing and the frozen-feature versus text-pathway deficit
- `MeasureBench` for numeric parsing conventions and measurement-readout positioning

The central claim is narrower than the 2026 probe paper:

> A deterministic classical-CV scaffold can expose geometry to a frozen VLM's text pathway without training a probe or LoRA.

The central limitation is also explicit:

> When the scaffold is wrong, the model often follows it.

## Current outputs

- [`output/appendix/MEASUREMENT_PRECONDITION.md`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\output\appendix\MEASUREMENT_PRECONDITION.md): classical measurement quality as a precondition, not the dependent variable
- [`output/vlm/summary.csv`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\output\vlm\summary.csv): frozen-VLM metrics by condition and model
- [`output/vlm/trust_under_error_summary.csv`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\output\vlm\trust_under_error_summary.csv): trust-under-error metrics aggregated by `|delta|`
- [`output/vlm/RESULTS.md`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\output\vlm\RESULTS.md): headline findings
- [`output/comparison/COMPARISON.md`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\output\comparison\COMPARISON.md): side-by-side interpretation against `2603.06459`

## Main scripts

- [`source code/run_counting_demo.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\run_counting_demo.py)
- [`source code/run_angle_demo.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\run_angle_demo.py)
- [`source code/run_research_suite.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\run_research_suite.py)
- [`source code/run_vlm_eval.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\run_vlm_eval.py)
- [`source code/trust_under_error.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\trust_under_error.py)
- [`source code/build_reports.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\build_reports.py)
- [`source code/vlm_adapters.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\vlm_adapters.py)

## Run

From `C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render`:

```powershell
python "source code\run_research_suite.py"
python "source code\run_vlm_eval.py" --task angle --model qwen2.5-vl-3b --conditions all
python "source code\trust_under_error.py"
python "source code\build_reports.py"
```

Useful bounded runs:

```powershell
python "source code\run_vlm_eval.py" --task angle --model smolvlm2-2.2b --conditions all --limit 8
python "source code\run_vlm_eval.py" --task counting --model qwen2.5-vl-3b --conditions raw,pixels,text,both --limit 12
```

## Environment

- Windows is supported directly.
- A 10 GB GPU can run `Qwen2.5-VL-3B-Instruct` in 4-bit in this repo.
- CPU and Ollama fallbacks remain available.
- Cache keys are `sha256(image_bytes + prompt + model_id)`, so reruns are effectively free once cached.

## Real-data hook

`data/real/` and [`source code/fetch_real_subsets.py`](C:\Users\USER\Documents\course_Translations\Computer_vision\final_project\measure_and_render\source%20code\fetch_real_subsets.py) stage a small FSC147 counting subset and a MeasureBench subset with attribution metadata. The current checkout includes a 12-image MeasureBench slice; FSC147 was rate-limited in this environment and is documented in `data/real/SOURCES.md`. These assets are auxiliary context for the repo and are not the current headline comparison table, which is based on the continuous-angle scaffold benchmark.
