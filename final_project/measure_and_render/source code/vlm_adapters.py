from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from common import VLM_OUTPUT_DIR, ensure_base_directories


MODEL_ALIASES = {
    "qwen2.5-vl-3b": {"kind": "hf", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "dtype": "float16", "quant": "4bit"},
    "qwen2.5-vl-7b": {"kind": "hf", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "dtype": "float16", "quant": "4bit"},
    "smolvlm2-2.2b": {"kind": "hf", "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "dtype": "float16", "quant": "none"},
    "phi-3.5-vision": {"kind": "hf", "model_id": "microsoft/Phi-3.5-vision-instruct", "dtype": "float16", "quant": "none"},
}


def create_data_uri(image_path: Path) -> tuple[str, str]:
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return mime_type, f"data:{mime_type};base64,{encoded}"


def build_cache_key(image_path: Path, prompt: str, model_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    digest.update(prompt.encode("utf-8"))
    digest.update(model_id.encode("utf-8"))
    return digest.hexdigest()


def parse_number(reply: str) -> float | None:
    if not reply:
        return None
    normalized = reply.replace(",", "")
    normalized = normalized.replace("≈", " ").replace("~", " ")
    normalized = normalized.replace("about", " ").replace("approximately", " ")
    time_matches = re.findall(r"(\d+):(\d+)", normalized)
    if time_matches:
        hours, minutes = time_matches[-1]
        return float(int(hours) * 60 + int(minutes))
    fraction_matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", normalized)
    numeric_matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", normalized)
    if not numeric_matches and not fraction_matches:
        return None
    if fraction_matches:
        last_fraction = fraction_matches[-1]
        try:
            numerator = float(last_fraction[0])
            denominator = float(last_fraction[1])
            if abs(denominator) > 1e-9:
                return numerator / denominator
        except ValueError:
            pass
    try:
        return float(numeric_matches[-1])
    except ValueError:
        return None


class VisionLanguageModelAdapter(ABC):
    def __init__(self, model_id: str, cache_dir: Path | None = None) -> None:
        ensure_base_directories()
        self.model_id = model_id
        self.cache_dir = (cache_dir or (VLM_OUTPUT_DIR / "cache")) / self.safe_model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_metadata: dict[str, Any] = {}

    @property
    def safe_model_name(self) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", self.model_id)

    def cache_path(self, image_path: Path, prompt: str) -> Path:
        return self.cache_dir / f"{build_cache_key(image_path, prompt, self.model_id)}.json"

    def answer(self, image_path: Path, prompt: str) -> str:
        cache_path = self.cache_path(image_path, prompt)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(payload["reply"])

        started = time.perf_counter()
        self.last_metadata = {}
        reply = self._answer_uncached(image_path, prompt)
        payload = {
            "model_id": self.model_id,
            "image_path": str(image_path),
            "prompt": prompt,
            "reply": reply,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "adapter": self.__class__.__name__,
        }
        payload.update(self.last_metadata)
        cache_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return reply

    def parse_number(self, reply: str) -> float | None:
        return parse_number(reply)

    @abstractmethod
    def _answer_uncached(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class OpenAIAdapter(VisionLanguageModelAdapter):
    def __init__(self, model_id: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        super().__init__(model_id)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def _answer_uncached(self, image_path: Path, prompt: str) -> str:
        _mime_type, data_uri = create_data_uri(image_path)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.0,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


class AnthropicAdapter(VisionLanguageModelAdapter):
    def __init__(self, model_id: str = "claude-3-5-sonnet-20241022", api_key: str | None = None) -> None:
        super().__init__(model_id)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    def _answer_uncached(self, image_path: Path, prompt: str) -> str:
        mime_type, data_uri = create_data_uri(image_path)
        encoded = data_uri.split(",", 1)[1]
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model_id,
                "max_tokens": 128,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}},
                        ],
                    }
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        chunks = [item.get("text", "") for item in payload.get("content", []) if item.get("type") == "text"]
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


class GeminiAdapter(VisionLanguageModelAdapter):
    def __init__(self, model_id: str = "gemini-1.5-flash", api_key: str | None = None) -> None:
        super().__init__(model_id)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    def _answer_uncached(self, image_path: Path, prompt: str) -> str:
        mime_type, data_uri = create_data_uri(image_path)
        encoded = data_uri.split(",", 1)[1]
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": encoded}},
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 128},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {payload}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()


class OllamaAdapter(VisionLanguageModelAdapter):
    def __init__(self, model_id: str = "qwen2.5vl:7b", host: str | None = None) -> None:
        super().__init__(model_id)
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def _answer_uncached(self, image_path: Path, prompt: str) -> str:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = requests.post(
            f"{self.host.rstrip('/')}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": self.model_id, "prompt": prompt, "images": [encoded], "stream": False, "options": {"temperature": 0}},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["response"]).strip()


class HFLocalAdapter(VisionLanguageModelAdapter):
    def __init__(self, model_id: str, dtype: str = "float16", quant: str = "none") -> None:
        super().__init__(model_id)
        self.dtype = dtype
        self.quant = quant
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def _resolve_dtype(self, torch_module):
        if self.dtype == "float16":
            return torch_module.float16
        if self.dtype == "bfloat16":
            return torch_module.bfloat16
        return None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            model_loader = AutoModelForImageTextToText
        except Exception:
            from transformers import AutoModelForVision2Seq  # type: ignore

            model_loader = AutoModelForVision2Seq

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        resolved_dtype = self._resolve_dtype(torch)
        if resolved_dtype is not None:
            load_kwargs["torch_dtype"] = resolved_dtype
        if self.quant == "4bit":
            load_kwargs["load_in_4bit"] = True
        if "qwen" in self.model_id.lower():
            load_kwargs["attn_implementation"] = "eager"

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        try:
            self._model = model_loader.from_pretrained(self.model_id, **load_kwargs)
        except Exception as exc:
            if self.quant != "4bit":
                raise
            load_kwargs.pop("load_in_4bit", None)
            self._model = model_loader.from_pretrained(self.model_id, **load_kwargs)
            print(f"Recovered from initial 4-bit load error for {self.model_id}: {exc}")
        self._torch = torch

    def _move_inputs(self, inputs: Any) -> tuple[Any, Any]:
        assert self._model is not None
        target_device = next(self._model.parameters()).device
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(target_device) if hasattr(value, "to") else value
        return moved, target_device

    def _answer_uncached(self, image_path: Path, prompt: str) -> str:
        self._ensure_loaded()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        processor = self._processor
        model = self._model
        torch_module = self._torch
        assert processor is not None
        assert model is not None
        assert torch_module is not None

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        moved_inputs, target_device = self._move_inputs(inputs)

        can_measure_cuda = getattr(torch_module.cuda, "is_available", lambda: False)() and str(target_device).startswith("cuda")
        if can_measure_cuda:
            torch_module.cuda.reset_peak_memory_stats(target_device)
        outputs = model.generate(**moved_inputs, max_new_tokens=96, do_sample=False)
        prompt_length = moved_inputs["input_ids"].shape[1] if "input_ids" in moved_inputs else 0
        generated = outputs[:, prompt_length:]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)

        metadata: dict[str, Any] = {
            "dtype": self.dtype,
            "quant": self.quant,
            "device": str(target_device),
        }
        if can_measure_cuda:
            peak_bytes = int(torch_module.cuda.max_memory_allocated(target_device))
            metadata["peak_allocated_gb"] = round(peak_bytes / (1024**3), 3)
        self.last_metadata = metadata
        return decoded[0].strip()


def build_adapter(model_name: str) -> VisionLanguageModelAdapter:
    if model_name.startswith("openai:"):
        return OpenAIAdapter(model_name.split(":", 1)[1])
    if model_name.startswith("anthropic:"):
        return AnthropicAdapter(model_name.split(":", 1)[1])
    if model_name.startswith("gemini:"):
        return GeminiAdapter(model_name.split(":", 1)[1])
    if model_name.startswith("ollama:"):
        return OllamaAdapter(model_name.split(":", 1)[1])
    if model_name.startswith("hf:"):
        return HFLocalAdapter(model_name.split(":", 1)[1])

    alias = MODEL_ALIASES.get(model_name)
    if alias and alias["kind"] == "hf":
        return HFLocalAdapter(alias["model_id"], dtype=str(alias.get("dtype", "float16")), quant=str(alias.get("quant", "none")))
    return HFLocalAdapter(model_name)
