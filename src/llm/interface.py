"""
Abstract LLM interface. All backends implement this contract.
Switch backends via LLM_BACKEND env var: mock | ollama | vllm
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Common interface for all LLM backends."""

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>…</think> chain-of-thought blocks emitted by reasoning models."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    @abstractmethod
    async def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Backend-specific completion implementation."""
        ...

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a single completion, stripping any <think> reasoning blocks."""
        return self._strip_think(
            await self._generate(prompt, system=system, temperature=temperature,
                                 max_tokens=max_tokens)
        )

    @abstractmethod
    async def generate_batch(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Generate completions for a list of prompts. Backends may parallelise."""
        ...

    async def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """Wrapper that instructs the model to return only JSON."""
        json_system = (system + "\n\nIMPORTANT: Respond with valid JSON only. "
                       "No markdown fences, no preamble, no explanation.").strip()
        return await self.generate(prompt, system=json_system, temperature=temperature)


def get_llm_client(backend: Optional[str] = None) -> BaseLLMClient:
    """
    Factory function. Returns the correct LLM client based on config.

    Usage:
        llm = get_llm_client()          # reads LLM_BACKEND from env
        llm = get_llm_client("mock")    # explicit override
    """
    from config.settings import settings

    backend = backend or settings.llm_backend
    backend = backend.lower().strip()

    if backend == "mock":
        from src.llm.mock_client import MockLLMClient
        logger.info("LLM backend: MockClient (no GPU required)")
        return MockLLMClient()

    elif backend == "ollama":
        from src.llm.ollama_client import OllamaClient
        logger.info(f"LLM backend: Ollama ({settings.ollama_model_name})")
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model_name,
        )

    elif backend == "vllm":
        from src.llm.vllm_client import VLLMClient
        logger.info(f"LLM backend: vLLM ({settings.vllm_model_name})")
        return VLLMClient(
            base_url=settings.vllm_base_url,
            model=settings.vllm_model_name,
        )

    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend}'. Choose from: mock, ollama, vllm"
        )
