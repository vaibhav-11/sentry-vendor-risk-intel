"""
Vendor Risk Intelligence — CLI runner.

Usage:
    # Mock mode (local, no GPU):
    python scripts/run_pipeline.py --company "Apple Inc" --backend mock

    # Ollama (local model):
    python scripts/run_pipeline.py --company "Microsoft" --backend ollama

    # vLLM on AMD MI300X:
    python scripts/run_pipeline.py --company "Tesla" --ticker TSLA --backend vllm

    # vLLM with explicit URL/model (e.g. Jupyter on AMD without .env):
    python scripts/run_pipeline.py --company "Apple Inc" --backend vllm \\
        --vllm-url http://localhost:8000/v1 \\
        --vllm-model ./models/Qwen2.5-14B-Instruct-GPTQ-Int4

    # Open result automatically:
    python scripts/run_pipeline.py --company "Apple Inc" --open
"""

import sys
import asyncio
import argparse
import logging
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Pre-flight (shared with generate_demo.py) ─────────────────────────────────

async def _check_vllm(base_url: str) -> tuple[bool, str]:
    import httpx
    root = base_url.rstrip("/").removesuffix("/v1")
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{root}/health")
            if r.status_code != 200:
                return False, f"HTTP {r.status_code} from {root}/health"
        except Exception as e:
            return False, f"Cannot reach vLLM at {root}: {e}"
        try:
            r     = await client.get(f"{root}/v1/models")
            ids   = [m["id"] for m in r.json().get("data", [])]
            label = ", ".join(ids) if ids else "(still loading)"
            return True, f"Serving: {label}"
        except Exception as e:
            return True, f"Health ok, /v1/models error: {e}"


async def _check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    import httpx
    root = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r    = await client.get(f"{root}/api/tags")
            tags = [m["name"] for m in r.json().get("models", [])]
            if model not in tags and not any(model.split(":")[0] in t for t in tags):
                return False, f"Model '{model}' not in Ollama. Available: {tags or '(none)'}"
            return True, f"Model '{model}' available"
        except Exception as e:
            return False, f"Cannot reach Ollama at {root}: {e}"


async def preflight_check(backend: str, vllm_url: str, ollama_url: str, ollama_model: str) -> bool:
    if backend == "mock":
        return True

    console.print(f"\n[dim]Pre-flight: checking {backend} backend…[/dim]")

    if backend == "vllm":
        ok, msg = await _check_vllm(vllm_url)
    elif backend == "ollama":
        ok, msg = await _check_ollama(ollama_url, ollama_model)
    else:
        console.print(f"[red]Unknown backend '{backend}'[/red]")
        return False

    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    console.print(f"  {icon} {backend}: {msg}")

    if not ok:
        if backend == "vllm":
            console.print(
                "\n[yellow]Start vLLM on AMD:[/yellow]\n"
                "  export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH\n"
                "  export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so:"
                "/opt/rocm/lib/librocsolver.so:/opt/rocm/lib/libhipsolver.so\n"
                "  python -m vllm.entrypoints.openai.api_server \\\n"
                "      --model ./models/Qwen2.5-14B-Instruct-GPTQ-Int4 \\\n"
                "      --dtype float16 --max-model-len 4096 \\\n"
                "      --gpu-memory-utilization 0.85 --port 8000 &"
            )
    return ok


def apply_env_overrides(backend, vllm_url, vllm_model, ollama_url, ollama_model):
    """Push CLI flags into env so pydantic-settings picks them up."""
    os.environ["LLM_BACKEND"] = backend
    if vllm_url:   os.environ["VLLM_BASE_URL"]    = vllm_url
    if vllm_model: os.environ["VLLM_MODEL_NAME"]   = vllm_model
    if ollama_url: os.environ["OLLAMA_BASE_URL"]   = ollama_url
    if ollama_model: os.environ["OLLAMA_MODEL_NAME"] = ollama_model


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    company: str,
    ticker: str | None,
    backend: str,
    vllm_url: str,
    vllm_model: str,
    ollama_url: str,
    ollama_model: str,
    open_browser: bool,
) -> None:
    apply_env_overrides(backend, vllm_url, vllm_model, ollama_url, ollama_model)

    from src.pipeline.workflow import run_pipeline
    from config.settings import settings

    # Resolve effective values (may come from .env if CLI flags were blank)
    effective_vllm_url   = os.environ.get("VLLM_BASE_URL",    settings.vllm_base_url)
    effective_ollama_url = os.environ.get("OLLAMA_BASE_URL",  settings.ollama_base_url)
    effective_ollama_mdl = os.environ.get("OLLAMA_MODEL_NAME",settings.ollama_model_name)

    ok = await preflight_check(backend, effective_vllm_url, effective_ollama_url, effective_ollama_mdl)
    if not ok:
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold blue]Vendor Risk Intelligence[/bold blue]\n"
        f"Target: [bold white]{company}[/bold white]  |  "
        f"Backend: [yellow]{backend}[/yellow]",
        border_style="blue"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline…", total=None)
        state = await run_pipeline(company, ticker, backend)
        progress.update(task, description="[green]Pipeline complete!")

    # Results table
    table = Table(title="Pipeline Summary", border_style="slate_blue1")
    table.add_column("Metric",  style="cyan",  no_wrap=True)
    table.add_column("Value",   style="white", no_wrap=True)
    table.add_row("Total Entities",  str(len(state.entities)))
    table.add_row("Risk Scores",     str(len(state.risk_scores)))
    table.add_row("Alerts",          str(len(state.alerts)))
    table.add_row("Pipeline Errors", str(len(state.errors)))
    table.add_row("LLM Backend",     backend)

    if state.graph_metrics:
        table.add_row("Graph Nodes",  str(state.graph_metrics.total_nodes))
        table.add_row("SPOFs",        str(len(state.graph_metrics.single_points_of_failure)))

    console.print(table)

    if state.errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for e in state.errors:
            console.print(f"  [yellow]•[/yellow] {e}")

    if state.dashboard_html:
        filename = f"{company.lower().replace(' ', '_')}_{state.run_id}.html"
        out_path = settings.output_dir / filename
        out_path.write_text(state.dashboard_html, encoding="utf-8")
        console.print(f"\n[green]✓ Dashboard saved:[/green] {out_path}")

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{out_path.resolve()}")
            console.print("[green]✓ Opened in browser[/green]")
    else:
        console.print("[red]✗ No dashboard generated — check errors above[/red]")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vendor Risk Intelligence Pipeline")
    parser.add_argument("--company",      required=True,      help="Target company name")
    parser.add_argument("--ticker",       default=None,       help="Stock ticker (optional)")
    parser.add_argument("--backend",      default="mock",     help="LLM backend: mock | ollama | vllm")
    parser.add_argument("--vllm-url",     default="",         help="vLLM base URL override")
    parser.add_argument("--vllm-model",   default="",         help="vLLM model name override")
    parser.add_argument("--ollama-url",   default="",         help="Ollama base URL override")
    parser.add_argument("--ollama-model", default="",         help="Ollama model name override")
    parser.add_argument("--open",         action="store_true",help="Open dashboard in browser after run")
    parser.add_argument("--verbose",      action="store_true",help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    asyncio.run(main(
        company=args.company,
        ticker=args.ticker,
        backend=args.backend,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        open_browser=args.open,
    ))