"""
One-command demo runner.
Runs the full pipeline for Apple Inc in mock mode and opens the HTML result.
Use this to verify the full stack works before switching to AMD/vLLM.

Usage:
    python scripts/generate_demo.py
    python scripts/generate_demo.py --company "Microsoft" --ticker MSFT
    python scripts/generate_demo.py --backend vllm          # on AMD
"""

import sys
import asyncio
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


async def run_demo(company: str, ticker: str | None, backend: str) -> None:
    from src.pipeline.workflow import run_pipeline
    from config.settings import settings

    console.print(Panel.fit(
        f"[bold]Vendor Risk Intel — Demo[/bold]\n"
        f"[dim]Company:[/dim] [cyan]{company}[/cyan]   "
        f"[dim]Backend:[/dim] [yellow]{backend}[/yellow]",
        border_style="cyan"
    ))

    console.print("\n[dim]Stage 1/5[/dim] Generating supply chain watchlist...")
    state = await run_pipeline(company, ticker, backend)

    console.print(f"[green]✓[/green] Watchlist: {len(state.entities)} entities across supply chain")
    console.print(f"[green]✓[/green] Footprints: {len(state.footprint_data)} entities with data")
    console.print(f"[green]✓[/green] Risk scores: {len(state.risk_scores)} entities scored")
    console.print(f"[green]✓[/green] Alerts generated: {len(state.alerts)}")

    if state.graph_metrics:
        gm = state.graph_metrics
        console.print(f"[green]✓[/green] Graph: {gm.total_nodes} nodes, {len(gm.single_points_of_failure)} SPOFs")

    # Save dashboard
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{company.lower().replace(' ', '_')}_demo_{ts}.html"
    out_path = settings.output_dir / filename

    if state.dashboard_html:
        out_path.write_text(state.dashboard_html, encoding="utf-8")
        console.print(f"\n[bold green]✓ Dashboard ready:[/bold green] {out_path}")
        console.print(f"[dim]Size: {len(state.dashboard_html):,} bytes[/dim]")

        # Open in browser
        try:
            webbrowser.open(f"file://{out_path.resolve()}")
            console.print("[green]✓ Opened in default browser[/green]")
        except Exception:
            console.print("[yellow]Could not auto-open browser — open the file manually[/yellow]")

        console.print(f"\n[bold]To run on AMD with a real LLM:[/bold]")
        console.print(f"  1. Push this repo to GitHub")
        console.print(f"  2. Clone on AMD, run: bash scripts/setup_amd.sh")
        console.print(f"  3. Run: python scripts/run_pipeline.py --company \"{company}\" --backend vllm --open")
    else:
        console.print("[red]✗ No dashboard generated — check pipeline errors[/red]")
        for err in state.errors:
            console.print(f"  [red]•[/red] {err}")


def main():
    parser = argparse.ArgumentParser(description="Run demo pipeline")
    parser.add_argument("--company", default="Apple Inc", help="Target company")
    parser.add_argument("--ticker",  default="AAPL",      help="Ticker symbol")
    parser.add_argument("--backend", default="mock",      help="mock|ollama|vllm")
    args = parser.parse_args()
    asyncio.run(run_demo(args.company, args.ticker, args.backend))


if __name__ == "__main__":
    main()
