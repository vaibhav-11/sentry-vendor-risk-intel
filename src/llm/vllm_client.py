"""
vLLM client for AMD MI300X.
vLLM exposes an OpenAI-compatible REST API on port 8000.
Start vLLM server with:
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \\
        --dtype float16 \\
        --max-model-len 8192 \\
        --gpu-memory-utilization 0.90
"""

import asyncio
import logging
from openai import AsyncOpenAI
from src.llm.interface import BaseLLMClient

logger = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    def __init__(self, base_url: str, model: str):
        self.model = model
        # vLLM OpenAI-compatible endpoint — no real API key needed
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise

    async def generate_batch(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Fire all prompts concurrently — MI300X handles the batching."""
        tasks = [
            self.generate(p, system=system, temperature=temperature, max_tokens=max_tokens)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
