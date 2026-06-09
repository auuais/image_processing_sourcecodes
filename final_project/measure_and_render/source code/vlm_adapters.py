from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VisionLanguageModelAdapter(ABC):
    @abstractmethod
    def answer(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class UnconfiguredAdapter(VisionLanguageModelAdapter):
    def answer(self, image_path: Path, prompt: str) -> str:
        raise RuntimeError(
            "No VLM adapter is configured yet. "
            "Connect an API or local VLM implementation here before running VLM experiments."
        )
