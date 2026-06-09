Real counting subset source: https://huggingface.co/datasets/isentropic/FSC147
Real measurement subset source: https://huggingface.co/datasets/FlagEval/MeasureBench
The fetch script writes deterministic local copies plus metadata under data/real/ when downloads succeed.

Fetch notes:
- A 12-image MeasureBench subset is present under `data/real/measurebench/`.
- FSC147 download was blocked in this environment by Hugging Face anonymous rate limiting (`429 Too Many Requests`), so the counting-side real subset is not populated in this checkout.
