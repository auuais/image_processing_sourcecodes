# Comparison to arXiv:2603.06459

This table is protocol-aligned, not dataset-identical. The 2026 paper reports hand-joint-angle MAE on its own datasets, while this repo reports a harder synthetic continuous-angle scaffold benchmark. The comparison is therefore about pathway behavior and training cost, not direct leaderboard replacement.

## Main readout

- Primary overlapping local model: `qwen2.5-vl-3b`.
- Our raw VLM MAE is 49.94 deg.
- Our best training-free scaffold MAE is 2.77 deg.
- The 2026 paper still wins on absolute MAE with trained probes or LoRA, but those methods require supervision and optimization that our scaffold avoids.

## Trust caveat

- At `|delta|=10 deg`, follow-rate is 94.79% for `pixels`, 80.73% for `text`, and 93.23% for `both`.
- That means the scaffold bridges the text-pathway deficit, but it does not automatically make the model skeptical of wrong measurements.

## Positioning

- `2603.06459` shows that frozen features know geometry but the text pathway under-reads them.
- This repo shows that a classical-CV render can expose that geometry to the frozen text pathway without training a new readout head.
- The honest limitation is trust: once the rendered measurement is wrong, the model often follows it.
