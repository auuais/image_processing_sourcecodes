# Model Notes

This file records the frozen VLMs actually exercised in this repo and the memory tier they fit.

## Hardware used

- GPU: `NVIDIA GeForce RTX 3080`
- VRAM budget available to this repo: `10 GB`
- OS: Windows
- Python: `3.9`

## Confirmed local runs

### `Qwen/Qwen2.5-VL-3B-Instruct`

- Adapter: `hf`
- Invocation alias: `qwen2.5-vl-3b`
- Dtype: `float16`
- Quantization: `4bit`
- Peak allocated GPU memory observed in cache metadata: `3.362 GB`
- Mean uncached latency from current cache log: `3.797 s/inference`
- Median uncached latency from current cache log: `1.525 s/inference`
- Memory note: fits on the 10 GB card only in the quantized path
- Study status: full synthetic angle run, `n=96`, all base and trust-under-error conditions
- Output location: `output/vlm/summary.csv`, `output/comparison/`, `output/vlm/RESULTS.md`

### `HuggingFaceTB/SmolVLM2-2.2B-Instruct`

- Adapter: `hf`
- Invocation alias: `smolvlm2-2.2b`
- Dtype: `float16`
- Quantization: `none`
- Peak allocated GPU memory observed in cache metadata: `4.606 GB`
- Mean uncached latency from current cache log: `3.002 s/inference`
- Median uncached latency from current cache log: `0.995 s/inference`
- Memory note: small enough for the same GPU tier and suitable as a fallback pilot model
- Study status: pilot angle run only; not used for headline claims

## Operational notes

- The adapter cache stores `latency_seconds` for every uncached call under `output/vlm/cache/<model>/`.
- New cache entries from local HF runs also store `device`, `dtype`, `quant`, and GPU `peak_allocated_gb` when CUDA reporting is available.
- `Qwen2.5-VL-7B-Instruct` remains wired in as an alias, but the current completed report is built around the 3B model because it overlaps the 2026 comparison paper and fits the local hardware budget.
