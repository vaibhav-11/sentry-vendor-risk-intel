"""
Vendor Risk Intelligence — CLI runner.

Usage:
    # Mock mode (local, no GPU):
    python scripts/run_pipeline.py --company "Apple Inc" --backend mock

    # Ollama (local model):
    python scripts/run_pipeline.py --company "Microsoft" --backend ollama

    # vLLM on AMD MI300X:
    python scripts/run_pipeline.py --company "Tesla" --ticker TSLA --backend vllm

    # Open result automatically:
    python scripts/run_pipeline.py --company "Apple Inc" --open
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

from src.pipeline.workflow import run_pipeline
from config.settings import settings

console = Console()
logging.basicConfig(level=logging.WARNING)     # Quiet by default
logger = logging.getLogger(__name__)


async def main(company: str, ticker: str | None, backend: str, open_browser: bool) -> None:
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
        task = progress.add_task("Running pipeline...", total=None)

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
        console.print(f"\n[green]✓ Dashboard saved:[/green] {out_path}")

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{out_path.resolve()}")
            console.print("[green]✓ Opened in browser[/green]")
    else:
        console.print("[red]✗ No dashboard generated — check errors above[/red]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vendor Risk Intelligence Pipeline")
    parser.add_argument("--company",  required=True,      help="Target company name")
    parser.add_argument("--ticker",   default=None,       help="Stock ticker (optional)")
    parser.add_argument("--backend",  default="mock",     help="LLM backend: mock|ollama|vllm")
    parser.add_argument("--open",     action="store_true", help="Open dashboard in browser after run")
    parser.add_argument("--verbose",  action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    asyncio.run(main(args.company, args.ticker, args.backend, args.open))
