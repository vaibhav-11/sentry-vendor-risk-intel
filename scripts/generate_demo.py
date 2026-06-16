"""
One-command demo runner.
Runs the full pipeline for Apple Inc in mock mode and opens the HTML result.
Use this to verify the full stack works before switching to AMD/vLLM.

Usage:
    python scripts/generate_demo.py
    python scripts/generate_demo.py --company "Microsoft" --ticker MSFT
    python scripts/generate_demo.py --backend vllm                    # on AMD
    python scripts/generate_demo.py --backend vllm --vllm-url http://localhost:8000/v1
    python scripts/generate_demo.py --backend ollama --ollama-model llama3.1:8b
"""

import sys
import time
import asyncio
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Pre-flight checks ─────────────────────────────────────────────────────────

async def check_vllm_health(base_url: str) -> tuple[bool, str]:
    """
    Ping the vLLM /health and /v1/models endpoints.
    Returns (ok, message).
    """
    import httpx

    # Strip /v1 suffix if present — health endpoint is at root
    root_url = base_url.rstrip("/").removesuffix("/v1")

    async with httpx.AsyncClient(timeout=5) as client:
        # 1. Health check
        try:
            r = await client.get(f"{root_url}/health")
            if r.status_code != 200:
                return False, f"Health endpoint returned HTTP {r.status_code}"
        except Exception as e:
            return False, f"Cannot reach vLLM at {root_url}: {e}"

        # 2. List models — confirm something is loaded
        try:
            r = await client.get(f"{root_url}/v1/models")
            models = r.json().get("data", [])
            if not models:
                return False, "vLLM is up but no models are loaded yet (still initialising?)"
            model_ids = [m["id"] for m in models]
            return True, f"Ready — serving: {', '.join(model_ids)}"
        except Exception as e:
            return False, f"vLLM health ok but /v1/models failed: {e}"


async def check_ollama_health(base_url: str, model_name: str) -> tuple[bool, str]:
    """Ping Ollama /api/tags and verify the model is available."""
    import httpx

    root_url = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{root_url}/api/tags")
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            if model_name not in tags:
                # Accept prefix match (e.g. "llama3.1" matches "llama3.1:8b")
                partial = [t for t in tags if model_name.split(":")[0] in t]
                if not partial:
                    return False, (
                        f"Model '{model_name}' not found in Ollama. "
                        f"Available: {tags or '(none)'}. "
                        f"Run: ollama pull {model_name}"
                    )
            return True, f"Ready — model '{model_name}' available"
        except Exception as e:
            return False, f"Cannot reach Ollama at {root_url}: {e}"


async def preflight(backend: str, vllm_url: str, ollama_url: str, ollama_model: str) -> bool:
    """
    Run pre-flight checks for the selected backend.
    Returns True if safe to proceed.
    """
    if backend == "mock":
        console.print("[dim]Backend: mock — no server required, skipping pre-flight[/dim]")
        return True

    console.print(f"\n[bold]Pre-flight check[/bold] — verifying [yellow]{backend}[/yellow] backend...")

    if backend == "vllm":
        ok, msg = await check_vllm_health(vllm_url)
    elif backend == "ollama":
        ok, msg = await check_ollama_health(ollama_url, ollama_model)
    else:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        return False

    if ok:
        console.print(f"  [green]✓[/green] {backend}: {msg}")
        return True
    else:
        console.print(f"  [red]✗[/red] {backend}: {msg}")
        console.print()

        if backend == "vllm":
            console.print("[yellow]To start vLLM on AMD MI300X:[/yellow]")
            console.print(
                "  export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH\n"
                "  export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so:"
                "/opt/rocm/lib/librocsolver.so:/opt/rocm/lib/libhipsolver.so\n"
                "  python -m vllm.entrypoints.openai.api_server \\\\\n"
                "      --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \\\\\n"
                "      --dtype float16 \\\\\n"
                "      --max-model-len 4096 \\\\\n"
                "      --gpu-memory-utilization 0.85 \\\\\n"
                "      --host 0.0.0.0 --port 8000 &"
            )
            console.print("\n  Then re-run this script once the server is ready.")
        elif backend == "ollama":
            console.print(f"[yellow]Start Ollama:[/yellow]  ollama serve")
            console.print(f"[yellow]Pull model:[/yellow]   ollama pull {ollama_model}")

        # Ask user if they want to fall back to mock
        console.print()
        try:
            ans = input(f"  Proceed with [bold]mock[/bold] backend instead? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans == "y":
            console.print("[dim]Falling back to mock backend[/dim]")
            return "mock"   # signal to caller to swap backend

        return False


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_demo(
    company: str,
    ticker: str | None,
    backend: str,
    vllm_url: str,
    vllm_model: str,
    ollama_url: str,
    ollama_model: str,
    no_browser: bool,
) -> None:
    # Total pipeline wall clock — measured from the top of the entry point so it
    # spans pre-flight, the LangGraph run, and dashboard write-out.
    _pipeline_t0 = time.perf_counter()

    # Patch settings before importing the pipeline so env overrides take effect
    # even when running in Jupyter / AMD without a .env file present.
    _apply_env_overrides(backend, vllm_url, vllm_model, ollama_url, ollama_model)

    from src.pipeline.workflow import run_pipeline
    from config.settings import settings

    # Pre-flight — may return "mock" as a string to signal fallback
    result = await preflight(backend, vllm_url, ollama_url, ollama_model)
    if result == "mock":
        backend = "mock"
    elif not result:
        sys.exit(1)

    # Background GPU VRAM monitor — only meaningful on the real vLLM/AMD path.
    # On every other backend (and if preflight downgraded vllm → mock) it stays
    # dormant: never started, so no thread, no logs/gpu_memory.log, no overhead.
    from src.utils.gpu_monitor import GPUMonitor
    monitor = GPUMonitor()
    if backend == "vllm":
        monitor.start()

    console.print(Panel.fit(
        f"[bold]Vendor Risk Intel — Demo[/bold]\n"
        f"[dim]Company:[/dim] [cyan]{company}[/cyan]   "
        f"[dim]Ticker:[/dim]  [cyan]{ticker or '—'}[/cyan]   "
        f"[dim]Backend:[/dim] [yellow]{backend}[/yellow]",
        border_style="cyan"
    ))

    # Show what LLM endpoint will be used
    _print_backend_summary(backend, vllm_url, vllm_model, ollama_url, ollama_model)

    console.print()

    t_start = datetime.utcnow()

    with console.status("[bold green]Running pipeline…[/bold green]") as status:
        def update(msg: str):
            status.update(f"[bold green]{msg}[/bold green]")

        update("Stage 1/5 — generating supply chain watchlist…")
        state = await run_pipeline(company, ticker, backend)

    elapsed = (datetime.utcnow() - t_start).total_seconds()

    # Results summary table
    table = Table(title="Pipeline Results", border_style="cyan", show_header=True)
    table.add_column("Stage",   style="cyan",  no_wrap=True)
    table.add_column("Output",  style="white")
    table.add_row("Watchlist",    f"{len(state.entities)} entities, {len(state.relationships)} relationships")
    table.add_row("Footprints",   f"{len(state.footprint_data)} entities with data")
    table.add_row("Risk Scoring", f"{len(state.risk_scores)} entities scored, {len(state.alerts)} alerts")

    if state.graph_metrics:
        gm = state.graph_metrics
        table.add_row(
            "Cascade Analysis",
            f"{gm.total_nodes} nodes, {len(gm.single_points_of_failure)} SPOFs, "
            f"density={gm.density:.3f}"
        )

    table.add_row("Dashboard",  f"{'Generated' if state.dashboard_html else 'FAILED'}")
    table.add_row("Elapsed",    f"{elapsed:.1f}s  |  backend={backend}")
    console.print(table)

    # Risk level breakdown
    if state.risk_scores:
        from src.models import RiskLevel
        counts = {l.value: 0 for l in RiskLevel}
        for s in state.risk_scores.values():
            counts[s.risk_level.value] += 1
        console.print(
            f"  Risk levels — "
            f"[red]Critical: {counts['critical']}[/red]  "
            f"[orange3]High: {counts['high']}[/orange3]  "
            f"[yellow]Medium: {counts['medium']}[/yellow]  "
            f"[green]Low: {counts['low']}[/green]"
        )

    # Errors / warnings
    if state.errors:
        console.print(f"\n[yellow]Pipeline warnings ({len(state.errors)}):[/yellow]")
        for err in state.errors:
            console.print(f"  [yellow]•[/yellow] {err}")

    # Save & open dashboard
    if state.dashboard_html:
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{company.lower().replace(' ', '_')}_{backend}_{ts}.html"
        out_path = settings.output_dir / filename
        out_path.write_text(state.dashboard_html, encoding="utf-8")
        size_kb  = len(state.dashboard_html) // 1024
        console.print(f"\n[bold green]✓ Dashboard:[/bold green] {out_path}  [dim]({size_kb} KB)[/dim]")

        if not no_browser:
            try:
                webbrowser.open(f"file://{out_path.resolve()}")
                console.print("[green]✓ Opened in browser[/green]")
            except Exception:
                console.print("[yellow]Could not auto-open browser — open the file manually[/yellow]")

        if backend != "mock":
            console.print(f"\n[bold green]✓ Real LLM inference complete![/bold green]  "
                          f"Narratives and alerts generated by {backend}.")
        else:
            console.print(
                "\n[dim]Running on mock backend — narratives are templated.\n"
                "To use the real LLM on AMD:\n"
                "  python scripts/generate_demo.py "
                f"--company \"{company}\" --backend vllm[/dim]"
            )

        # ── Run metrics — latency + token usage summary ────────────────────
        monitor.stop()
        from src.llm.metrics import set_pipeline_wall_seconds, print_run_metrics
        set_pipeline_wall_seconds(time.perf_counter() - _pipeline_t0)
        console.print()
        print_run_metrics(monitor)
    else:
        monitor.stop()
        console.print("[red]✗ No dashboard generated — check errors above[/red]")
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_env_overrides(
    backend: str,
    vllm_url: str,
    vllm_model: str,
    ollama_url: str,
    ollama_model: str,
) -> None:
    """
    Write CLI overrides into os.environ BEFORE settings is imported by the
    pipeline. This means --vllm-url / --ollama-model work even without a .env.
    """
    import os
    os.environ.setdefault("LLM_BACKEND", backend)
    os.environ["LLM_BACKEND"] = backend

    if vllm_url:
        os.environ["VLLM_BASE_URL"]   = vllm_url
    if vllm_model:
        os.environ["VLLM_MODEL_NAME"] = vllm_model
    if ollama_url:
        os.environ["OLLAMA_BASE_URL"]   = ollama_url
    if ollama_model:
        os.environ["OLLAMA_MODEL_NAME"] = ollama_model


def _print_backend_summary(
    backend: str,
    vllm_url: str,
    vllm_model: str,
    ollama_url: str,
    ollama_model: str,
) -> None:
    if backend == "mock":
        console.print("  [dim]LLM: mock client (no GPU, templated responses)[/dim]")
    elif backend == "vllm":
        console.print(f"  [dim]LLM: vLLM @ {vllm_url}  model={vllm_model}[/dim]")
    elif backend == "ollama":
        console.print(f"  [dim]LLM: Ollama @ {ollama_url}  model={ollama_model}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vendor Risk Intel — demo pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mock (no GPU):
  python scripts/generate_demo.py

  # AMD MI300X with vLLM:
  python scripts/generate_demo.py --backend vllm

  # Override vLLM URL/model explicitly:
  python scripts/generate_demo.py --backend vllm \\
      --vllm-url http://localhost:8000/v1 \\
      --vllm-model ./models/Qwen2.5-14B-Instruct-GPTQ-Int4

  # Ollama local model:
  python scripts/generate_demo.py --backend ollama --ollama-model llama3.1:8b

  # Different company:
  python scripts/generate_demo.py --company "Microsoft" --ticker MSFT --backend vllm
        """,
    )
    parser.add_argument("--company",      default="Apple Inc",        help="Target company name")
    parser.add_argument("--ticker",       default="AAPL",             help="Stock ticker (optional)")
    parser.add_argument("--backend",      default="mock",             help="LLM backend: mock | ollama | vllm")

    # vLLM overrides (fall back to .env / settings defaults if not provided)
    parser.add_argument("--vllm-url",     default="",                 help="vLLM base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--vllm-model",   default="",                 help="vLLM model name/path")

    # Ollama overrides
    parser.add_argument("--ollama-url",   default="",                 help="Ollama base URL")
    parser.add_argument("--ollama-model", default="",                 help="Ollama model name")

    parser.add_argument("--no-browser",   action="store_true",        help="Don't auto-open browser")
    parser.add_argument("--verbose",      action="store_true",        help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, force=True)

    asyncio.run(run_demo(
        company=args.company,
        ticker=args.ticker,
        backend=args.backend,
        vllm_url=args.vllm_url or "",
        vllm_model=args.vllm_model or "",
        ollama_url=args.ollama_url or "",
        ollama_model=args.ollama_model or "",
        no_browser=args.no_browser,
    ))


if __name__ == "__main__":
    main()