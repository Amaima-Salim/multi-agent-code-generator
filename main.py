import sys

# Windows terminals default to cp1252 which can't render emoji.
# Force UTF-8 so Rich output works correctly.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(
    name="ai-dev-team-v2",
    help="AI Dev Team v2 — LangGraph · ChromaDB · Tavily · Multi-LLM",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()

EXAMPLES = [
    "Build a REST API for a todo app with FastAPI and SQLite",
    "Create a CLI tool that converts CSV files to JSON with filtering",
    "Build a URL shortener with FastAPI, SQLite, and click analytics",
    "Create a real-time chat server using FastAPI WebSockets",
]


@app.command()
def generate(
    requirement: str = typer.Argument(None, help="Requirement to implement"),
):
    """Run the full AI Dev Team pipeline (PM → Architect → Developer → QA → Reviewer)."""
    _check_env()

    if not requirement:
        _print_welcome()
        requirement = Prompt.ask(
            "\n[bold yellow]Enter your requirement[/bold yellow]",
            default=EXAMPLES[0],
        )

    if not requirement.strip():
        console.print("[red]Error: requirement cannot be empty[/red]")
        raise typer.Exit(1)

    from graph.pipeline import build_pipeline
    from graph.state import create_initial_state
    import uuid

    project_id = str(uuid.uuid4())[:8]
    pipeline = build_pipeline()
    initial = create_initial_state(requirement.strip(), project_id)

    console.print(Panel(
        f"[bold white]AI Dev Team v2[/bold white]\n\n"
        f"[cyan]Project ID:[/cyan]   {project_id}\n"
        f"[cyan]Requirement:[/cyan]  {requirement.strip()}\n\n"
        f"[dim]Stack: LangGraph · ChromaDB · Tavily · Multi-LLM[/dim]",
        title="[bold blue]🚀 Starting LangGraph Pipeline[/bold blue]",
        border_style="blue",
    ))

    final_state = None
    try:
        for snapshot in pipeline.stream(initial, stream_mode="values"):
            final_state = snapshot
    except Exception as exc:
        console.print(f"[bold red]Pipeline error: {exc}[/bold red]")
        raise typer.Exit(1)

    if final_state and final_state.get("status") == "completed":
        console.print(f"\n[bold green]✅ Done! Output:[/bold green] {final_state.get('output_path')}")

        # Show enrichment summary
        hits = final_state.get("memory_hits", [])
        searches = final_state.get("search_queries", [])
        if hits:
            console.print(f"[dim]🧠 ChromaDB: {len(hits)} memory hits used[/dim]")
        if searches:
            console.print(f"[dim]🔍 Tavily: {len(searches)} web searches made[/dim]")

        sys.exit(0)
    else:
        sys.exit(1)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    """Launch the web UI."""
    _check_env()
    import uvicorn

    console.print(Panel(
        f"[bold cyan]Web UI starting…[/bold cyan]\n\n"
        f"Open [bold]http://{host}:{port}[/bold] in your browser",
        border_style="cyan",
    ))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


@app.command()
def memory():
    """Show ChromaDB memory statistics."""
    _check_env()
    from memory.chroma import MemoryManager

    stats = MemoryManager().stats()
    console.print(Panel(
        f"[bold]ChromaDB Memory[/bold]\n\n"
        f"[cyan]QA Issues stored:[/cyan]       {stats['qa_issues']}\n"
        f"[cyan]Arch Decisions stored:[/cyan]  {stats['arch_decisions']}\n\n"
        f"[dim]Grows with each project run — QA agent learns from past issues.[/dim]",
        border_style="cyan",
        title="🧠 Memory Stats",
    ))


def _check_env() -> None:
    import os
    from pathlib import Path

    any_key = any(os.environ.get(k) for k in (
        "XAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY", "GEMINI_API_KEY",
    ))
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists() and not any_key:
        console.print(Panel(
            "[bold red]Setup required[/bold red]\n\n"
            "Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add an API key:\n\n"
            "  [bold]cp .env.example .env[/bold]\n\n"
            "Free options:\n"
            "  XAI_API_KEY    — console.x.ai\n"
            "  GROQ_API_KEY   — console.groq.com\n"
            "  GEMINI_API_KEY — aistudio.google.com",
            border_style="red",
        ))
        raise typer.Exit(1)


def _print_welcome() -> None:
    from config import settings

    provider_status = f"[green]✓ {settings.active_provider} / {settings.active_model}[/green]"
    tavily_status = "[green]✓ Web search enabled[/green]" if settings.has_tavily else "[dim]✗ not set (skipped)[/dim]"

    console.print(Panel(
        "[bold cyan]AI Dev Team v2[/bold cyan]\n\n"
        "Multi-agent code generator with:\n"
        "  [violet]📊 LangGraph[/violet]  — typed state machine orchestration\n"
        "  [green]🧠 ChromaDB[/green]   — persistent memory across runs\n"
        "  [yellow]🔍 Tavily[/yellow]    — live web search for the Developer agent\n"
        "  [blue]🤖 Multi-LLM[/blue]  — best model per agent\n\n"
        f"  LLM:     {provider_status}\n"
        f"  Tavily:  {tavily_status}\n\n"
        "[dim]Pipeline: PM → Architect → Developer → QA → Reviewer → Save[/dim]",
        title="[bold blue]🚀 Welcome[/bold blue]",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
