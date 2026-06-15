"""
Ollama client — useful if running a local GPU on Mac (M1/M2/M3 via Metal)
or as a fallback on AMD before vLLM is configured.
"""

import asyncio
import logging
import httpx
from src.llm.interface import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str, model: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as e:
                logger.error(f"Ollama generation error: {e}")
                raise

    async def generate_batch(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> list[str]:
        # Ollama handles one request at a time; run concurrently via asyncio
        tasks = [
            self.generate(p, system=system, temperature=temperature, max_tokens=max_tokens)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
