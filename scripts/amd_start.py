#!/usr/bin/env python3
"""
scripts/amd_start.py — AMD MI300X session startup helper.

Run this at the START of every AMD Jupyter session. It:
  1. Sets the required ROCm env vars in the current process
  2. Verifies the GPU is visible
  3. Starts vLLM in a background subprocess (if not already running)
  4. Waits for it to be ready
  5. Prints the commands to run the demo pipeline

Usage:
    python scripts/amd_start.py
    python scripts/amd_start.py --no-vllm   # just set env vars, skip server launch
    python scripts/amd_start.py --model ./models/Llama-3.1-8B-Instruct
"""

import os
import sys
import asyncio
import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── ROCm env vars required for vLLM on ROCm 7.0 ──────────────────────────────

ROCM_LIB_PATHS = [
    "/opt/rocm/lib",
    "/opt/rocm/lib64",
    "/opt/rocm/hipblas/lib",
    "/opt/rocm/rocblas/lib",
]

PRELOAD_LIBS = [
    "/opt/rocm/lib/libhsa-runtime64.so",
    "/opt/rocm/lib/librocsolver.so",
    "/opt/rocm/lib/libhipsolver.so",
]

DEFAULT_MODEL = "./models/Qwen2.5-14B-Instruct-GPTQ-Int4"
VLLM_PORT     = 8000
VLLM_URL      = f"http://localhost:{VLLM_PORT}"


def set_rocm_env() -> None:
    """Set ROCm library paths in the current process environment."""
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths   = ":".join(ROCM_LIB_PATHS)
    os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{existing_ld}".strip(":")

    # Only preload libs that actually exist on this machine
    existing_libs = [lib for lib in PRELOAD_LIBS if Path(lib).exists()]
    if existing_libs:
        existing_preload = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = ":".join(existing_libs + ([existing_preload] if existing_preload else []))

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"   # MI300X gfx940/941 hint for some kernels

    print("✓ ROCm env vars set")
    if existing_libs:
        print(f"  LD_PRELOAD: {', '.join(Path(l).name for l in existing_libs)}")
    else:
        print("  LD_PRELOAD: (no libs found at expected paths — check ROCm installation)")
    print(f"  LD_LIBRARY_PATH: ...{':'.join(ROCM_LIB_PATHS)}")


def check_gpu() -> bool:
    """Verify GPU is visible via rocm-smi or torch."""
    print("\nChecking GPU...")
    # Try rocm-smi first
    result = subprocess.run(
        ["rocm-smi", "--showproductname"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        for line in lines[:5]:
            print(f"  {line}")
        return True

    # Fall back to torch
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {name}  VRAM: {vram:.1f} GB")
            return True
        else:
            print("  ⚠ torch.cuda.is_available() = False — ROCm may not be configured")
            return False
    except ImportError:
        print("  ⚠ PyTorch not installed — cannot verify GPU")
        return False


async def is_vllm_running() -> bool:
    """Check if vLLM is already running on the expected port."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{VLLM_URL}/health")
            return r.status_code == 200
    except Exception:
        return False


def start_vllm_background(model: str) -> subprocess.Popen:
    """Launch vLLM as a background subprocess with the correct env."""
    env = os.environ.copy()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", "float16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
    ]

    log_path = Path("vllm_server.log")
    log_file = open(log_path, "w")
    proc     = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"\n  vLLM started (PID {proc.pid})")
    print(f"  Logs: tail -f {log_path.resolve()}")
    return proc


async def wait_for_vllm(timeout_seconds: int = 300) -> bool:
    """Poll until vLLM is ready or timeout is reached."""
    import httpx
    print(f"\nWaiting for vLLM to be ready (timeout {timeout_seconds}s)...", end="", flush=True)
    start = time.time()

    async with httpx.AsyncClient(timeout=3) as client:
        while time.time() - start < timeout_seconds:
            try:
                r = await client.get(f"{VLLM_URL}/health")
                if r.status_code == 200:
                    # Also check /v1/models to confirm model is loaded
                    r2    = await client.get(f"{VLLM_URL}/v1/models")
                    ids   = [m["id"] for m in r2.json().get("data", [])]
                    if ids:
                        print(f"\n✓ vLLM ready — model: {ids[0]}")
                        return True
            except Exception:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(5)

    print(f"\n✗ vLLM did not become ready within {timeout_seconds}s")
    print("  Check logs: tail -f vllm_server.log")
    return False


def print_next_steps(model: str) -> None:
    print("\n" + "=" * 60)
    print(" Environment ready. Next steps:")
    print("=" * 60)
    print()
    print("  # Quick demo (Apple Inc, real LLM):")
    print("  python scripts/generate_demo.py --backend vllm")
    print()
    print("  # Full CLI run:")
    print(f"  python scripts/run_pipeline.py --company 'Apple Inc' --ticker AAPL --backend vllm --open")
    print()
    print("  # In Jupyter — change BACKEND to 'vllm' in notebook cell 1:")
    print("  BACKEND = 'vllm'")
    print()
    print(f"  vLLM API:  {VLLM_URL}/v1")
    print(f"  Model:     {model}")
    print()


async def run(model: str, skip_vllm: bool, timeout: int) -> None:
    set_rocm_env()
    check_gpu()

    if skip_vllm:
        print("\n--no-vllm specified — skipping server launch")
        print_next_steps(model)
        return

    already_running = await is_vllm_running()
    if already_running:
        print("\n✓ vLLM is already running on port", VLLM_PORT)
        print_next_steps(model)
        return

    model_path = Path(model)
    if not model_path.exists():
        print(f"\n⚠ Model not found at '{model}'")
        print("  Run setup first: bash scripts/setup_amd.sh")
        print("  Or specify with: python scripts/amd_start.py --model <path>")
        print("\n  Continuing with model path as-is (HuggingFace Hub ID may work)...")

    start_vllm_background(model)
    ready = await wait_for_vllm(timeout)

    if ready:
        print_next_steps(model)
    else:
        print("\n✗ Setup incomplete — check vllm_server.log for errors")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AMD MI300X session startup")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Model path or HuggingFace ID")
    parser.add_argument("--no-vllm", action="store_true",    help="Only set env vars, don't start vLLM")
    parser.add_argument("--timeout", type=int, default=300,  help="Seconds to wait for vLLM (default 300)")
    args = parser.parse_args()

    asyncio.run(run(args.model, args.no_vllm, args.timeout))


if __name__ == "__main__":
    main()