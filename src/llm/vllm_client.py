"""
vLLM client for AMD MI300X.
vLLM exposes an OpenAI-compatible REST API on port 8000.

Start vLLM server with:
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
    export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so:/opt/rocm/lib/librocsolver.so:/opt/rocm/lib/libhipsolver.so
    python -m vllm.entrypoints.openai.api_server \\
        --model ./models/Qwen2.5-14B-Instruct-GPTQ-Int4 \\
        --dtype float16 \\
        --max-model-len 4096 \\
        --gpu-memory-utilization 0.85 \\
        --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from src.llm.interface import BaseLLMClient
import re

logger = logging.getLogger(__name__)

# MI300X can handle large batches well; individual requests may still be slow
# on first token due to KV cache warmup — give generous timeouts.
REQUEST_TIMEOUT   = 120   # seconds per single generation
BATCH_CONCURRENCY = 8     # parallel requests to vLLM; it batches internally


class VLLMClient(BaseLLMClient):
    def __init__(self, base_url: str, model: str):
        self.model    = model
        self.base_url = base_url
        # vLLM OpenAI-compatible endpoint — no real API key needed
        self.client   = AsyncOpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=2,
        )

    async def _resolve_model(self) -> str:
        """
        If self.model is a local path like ./models/Qwen... the server may
        register it under a different ID. Resolve the actual model ID from
        the /v1/models endpoint so completions don't 404.
        """
        try:
            models = await self.client.models.list()
            ids = [m.id for m in models.data]
            if not ids:
                logger.warning("vLLM returned no models — server may still be loading")
                return self.model
            # Exact match first
            if self.model in ids:
                return self.model
            # Path contains the model name somewhere
            for mid in ids:
                if any(part in mid for part in self.model.replace("./", "").split("/")):
                    logger.info(f"vLLM model resolved: '{self.model}' → '{mid}'")
                    return mid
            # Just use whatever is loaded
            logger.info(f"Using first available model: {ids[0]}")
            return ids[0]
        except Exception as e:
            logger.warning(f"Could not resolve model ID from vLLM: {e}. Using '{self.model}'")
            return self.model

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        model = await self._resolve_model()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            logger.debug(
                f"vLLM generate: {len(text)} chars, "
                f"tokens_in={response.usage.prompt_tokens}, "
                f"tokens_out={response.usage.completion_tokens}"
            )
            return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        except APIConnectionError as e:
            msg = (
                f"Cannot connect to vLLM at {self.base_url}. "
                f"Is the server running? ({e})"
            )
            logger.error(msg)
            raise ConnectionError(msg) from e

        except APITimeoutError as e:
            msg = (
                f"vLLM request timed out after {REQUEST_TIMEOUT}s. "
                f"The model may still be loading, or the prompt is too long. ({e})"
            )
            logger.error(msg)
            raise TimeoutError(msg) from e

        except RateLimitError as e:
            logger.warning(f"vLLM rate limit hit — waiting 2s and retrying: {e}")
            await asyncio.sleep(2)
            return await self.generate(prompt, system=system,
                                        temperature=temperature, max_tokens=max_tokens)

        except Exception as e:
            logger.error(f"vLLM generation error: {type(e).__name__}: {e}")
            raise

    async def generate_batch(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> list[str]:
        """
        Fire all prompts concurrently up to BATCH_CONCURRENCY at a time.
        The MI300X handles the GPU-side batching internally via continuous batching.
        """
        semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

        async def bounded(p: str) -> str:
            async with semaphore:
                return await self.generate(
                    p, system=system, temperature=temperature, max_tokens=max_tokens
                )

        logger.info(f"vLLM batch: {len(prompts)} prompts, concurrency={BATCH_CONCURRENCY}")
        results = await asyncio.gather(
            *[bounded(p) for p in prompts],
            return_exceptions=True,
        )

        # Surface exceptions as error strings rather than crashing the whole batch
        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"vLLM batch item {i} failed: {r}")
                output.append(f"[LLM ERROR: {r}]")
            else:
                output.append(r)
        return output