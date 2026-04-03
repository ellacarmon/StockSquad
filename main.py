#!/usr/bin/env python3
"""
StockSquad CLI - Multi-Agent Stock Research System

A command-line interface for analyzing stocks using a squad of specialized AI agents.
"""

import sys
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.markdown import Markdown

from config import get_settings
from agents.orchestrator import OrchestratorAgent
from agents.chat_agent import ChatAgent
from memory.long_term import LongTermMemory

# Initialize Typer app and Rich console
app = typer.Typer(
    name="stocksquad",
    help="Multi-Agent Stock Research System",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker symbol (e.g., AAPL, MSFT, NVDA)"),
    period: str = typer.Option("1y", help="Historical data period (1mo, 3mo, 6mo, 1y, 2y, 5y)"),
    show_data: bool = typer.Option(False, "--show-data", help="Show raw data collected"),
    save_report: bool = typer.Option(False, "--save", help="Save report to file"),
):
    """
    Analyze a stock ticker using the StockSquad agent system.

    This command orchestrates multiple AI agents to gather and analyze stock data,
    producing a comprehensive research report.
    """
    ticker = ticker.upper()

    # Display header
    console.print(Panel.fit(
        f"[bold cyan]StockSquad Analysis[/bold cyan]\n"
        f"Ticker: [yellow]{ticker}[/yellow] | Period: [yellow]{period}[/yellow]",
        border_style="cyan"
    ))

    try:
        # Verify configuration
        settings = get_settings()
        console.print("[dim]Configuration loaded successfully[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to load configuration")
        console.print(f"[red]{str(e)}[/red]")
        console.print("\n[yellow]Please ensure .env file exists with required Azure credentials[/yellow]")
        console.print("[dim]See .env.example for template[/dim]")
        raise typer.Exit(code=1)

    # Initialize orchestrator
    console.print("\n[cyan]Initializing agent system...[/cyan]")
    orchestrator = OrchestratorAgent()

    try:
        # Run analysis
        with console.status(f"[bold green]Analyzing {ticker}...", spinner="dots"):
            result = orchestrator.analyze_stock(ticker=ticker, period=period)

        if not result.get("success", False):
            console.print(f"\n[bold red]Analysis failed:[/bold red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(code=1)

        # Display results
        console.print("\n[bold green]Analysis complete![/bold green]")

        # Show statistics
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_row("Execution Time", f"{result.get('execution_time_seconds', 0):.2f}s")
        stats_table.add_row("Past Analyses Found", str(result.get('past_analyses_found', 0)))
        stats_table.add_row("Document ID", result.get('document_id', 'N/A'))
        console.print(Panel(stats_table, title="[cyan]Statistics[/cyan]", border_style="cyan"))

        # Display final report
        final_report = result.get("final_report", "No report generated")
        console.print("\n" + "=" * 80)
        console.print(Markdown(final_report))
        console.print("=" * 80)

        # Show raw data if requested
        if show_data:
            session_data = result.get("session_summary", {})
            scratchpad = session_data.get("scratchpad", {})
            if scratchpad:
                console.print("\n[bold cyan]Raw Data Collected:[/bold cyan]")
                for key, value in scratchpad.items():
                    console.print(f"\n[yellow]{key}:[/yellow]")
                    console.print(value.get("value", {}))

        # Save report if requested
        if save_report:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{ticker}_{result.get('timestamp', 'unknown')[:10]}.md"

            with open(output_file, "w") as f:
                f.write(f"# StockSquad Analysis: {ticker}\n\n")
                f.write(f"**Date:** {result.get('timestamp', 'N/A')}\n")
                f.write(f"**Period:** {period}\n")
                f.write(f"**Execution Time:** {result.get('execution_time_seconds', 0):.2f}s\n\n")
                f.write("---\n\n")
                f.write(final_report)

            console.print(f"\n[green]Report saved to:[/green] {output_file}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        raise typer.Exit(code=130)

    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)

    finally:
        # Cleanup
        try:
            orchestrator.cleanup()
        except:
            pass


@app.command()
def history(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    limit: int = typer.Option(5, help="Number of past analyses to show"),
):
    """
    View past analyses for a ticker from long-term memory.
    """
    ticker = ticker.upper()

    console.print(f"\n[cyan]Retrieving past analyses for {ticker}...[/cyan]\n")

    try:
        memory = LongTermMemory()
        analyses = memory.retrieve_past_analyses(ticker=ticker, limit=limit)

        if not analyses:
            console.print(f"[yellow]No past analyses found for {ticker}[/yellow]")
            return

        console.print(f"[green]Found {len(analyses)} past analysis(es)[/green]\n")

        for i, analysis in enumerate(analyses, 1):
            timestamp = analysis.get("timestamp", "Unknown")
            summary = analysis.get("summary", "No summary available")
            doc_id = analysis.get("metadata", {}).get("timestamp", "")

            panel = Panel(
                f"[dim]{summary}[/dim]\n\n"
                f"[cyan]Document ID:[/cyan] {doc_id[:50]}...",
                title=f"[yellow]Analysis #{i}[/yellow] - {timestamp[:10]}",
                border_style="blue",
            )
            console.print(panel)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def stats():
    """
    Show statistics about the stored analyses in long-term memory.
    """
    console.print("\n[cyan]Retrieving memory statistics...[/cyan]\n")

    try:
        memory = LongTermMemory()
        stats = memory.get_collection_stats()

        table = Table(title="Long-Term Memory Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Analyses", str(stats.get("total_analyses", 0)))
        table.add_row("Collection Name", stats.get("collection_name", "N/A"))
        table.add_row("Storage Path", str(stats.get("storage_path", "N/A")))

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def config():
    """
    Display current configuration settings (without sensitive data).
    """
    console.print("\n[cyan]Current Configuration:[/cyan]\n")

    try:
        settings = get_settings()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")

        # Show non-sensitive settings
        table.add_row("Azure OpenAI Endpoint", settings.azure_openai_endpoint)
        table.add_row("Chat Model", settings.azure_openai_deployment_name)
        table.add_row("Embedding Model", settings.azure_openai_embedding_deployment_name)
        table.add_row("ChromaDB Path", str(settings.chroma_db_path))
        table.add_row("Log Level", settings.log_level)

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("\n[yellow]Configuration not loaded. Please check .env file.[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def ui(
    port: int = typer.Option(8000, help="Port to run the UI server on")
):
    """
    Start the StockSquad Web UI (React + FastAPI).
    """
    console.print(Panel.fit(
        f"[bold cyan]Starting StockSquad UI[/bold cyan]\n"
        f"Server running on [green]http://localhost:{port}[/green]",
        border_style="cyan"
    ))
    try:
        import uvicorn
        uvicorn.run("ui.api:app", host="127.0.0.1", port=port, reload=True, reload_dirs=["ui"])
    except Exception as e:
        console.print(f"[bold red]Error starting UI:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def chat(
    ticker: str = typer.Argument(None, help="Stock ticker to chat about (optional)"),
    doc_id: str = typer.Option(None, "--doc-id", help="Specific analysis document ID to discuss"),
    web_search: bool = typer.Option(False, "--web", help="Enable web search for additional context"),
):
    """
    Interactive chat about stock analysis reports.

    Ask questions, get clarifications, and discuss the analysis findings.
    Optionally enable web search for latest market information.
    """
    console.print(Panel.fit(
        "[bold cyan]StockSquad Chat[/bold cyan]\n"
        f"{'Ticker: ' + ticker.upper() if ticker else 'General discussion'}\n"
        f"{'Web search: ENABLED' if web_search else 'Web search: DISABLED (use --web to enable)'}",
        border_style="cyan"
    ))

    try:
        # Initialize chat agent
        chat_agent = ChatAgent(web_search_enabled=web_search)

        # If doc_id provided, load and show summary
        if doc_id:
            console.print(f"\n[cyan]Loading analysis document: {doc_id[:20]}...[/cyan]")
        elif ticker:
            console.print(f"\n[cyan]Loading recent analyses for {ticker.upper()}...[/cyan]")

        console.print("\n[dim]Type your questions below. Type 'exit' or 'quit' to end the chat.[/dim]")
        console.print("[dim]Type 'clear' to clear conversation history.[/dim]")
        console.print("[dim]Type 'web:' before your question to enable web search for that query.[/dim]\n")

        # Chat loop
        while True:
            try:
                # Get user input
                user_input = typer.prompt("\n[You]", prompt_suffix=" ")

                if not user_input.strip():
                    continue

                # Check for commands
                if user_input.lower() in ['exit', 'quit']:
                    console.print("\n[yellow]Ending chat session. Goodbye![/yellow]")
                    break

                if user_input.lower() == 'clear':
                    chat_agent.clear_history()
                    console.print("[green]Conversation history cleared.[/green]")
                    continue

                # Check for web search prefix
                use_web = web_search
                question = user_input
                if user_input.lower().startswith('web:'):
                    use_web = True
                    question = user_input[4:].strip()
                    console.print("[dim]Web search enabled for this query...[/dim]")

                # Process the question
                with console.status("[bold green]Thinking...", spinner="dots"):
                    result = chat_agent.chat(
                        user_message=question,
                        ticker=ticker,
                        doc_id=doc_id,
                        include_web_search=use_web
                    )

                # Display response
                if result.get("error"):
                    console.print(f"\n[bold red]Error:[/bold red] {result.get('response')}")
                else:
                    console.print(f"\n[bold cyan][ChatAgent][/bold cyan]")
                    console.print(Markdown(result["response"]))

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Chat interrupted. Goodbye![/yellow]")
                break

            except EOFError:
                console.print("\n\n[yellow]Chat ended. Goodbye![/yellow]")
                break

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)

    finally:
        # Cleanup
        try:
            chat_agent.cleanup()
        except:
            pass


@app.command()
def version():
    """
    Show StockSquad version information.
    """
    console.print(Panel.fit(
        "[bold cyan]StockSquad[/bold cyan] v0.1.0\n"
        "[dim]Multi-Agent Stock Research System[/dim]\n"
        "[dim]Powered by Azure AI Foundry[/dim]",
        border_style="cyan"
    ))


def main():
    """Main entry point."""
    try:
        app()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
