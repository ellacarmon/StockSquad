"""
Telegram Screener Command Handler
Handles /screen commands for stock screening.
"""

import asyncio
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from tools.prebuilt_screens import PrebuiltScreens
from tools.batch_analyzer import BatchAnalyzer
from tools.stock_universe import StockUniverse


class ScreenerHandler:
    """Handles stock screening commands for Telegram bot."""

    def __init__(self):
        """Initialize screener handler."""
        self.screens = PrebuiltScreens()
        self.analyzer = BatchAnalyzer(max_workers=5)
        self.universe = StockUniverse()

    async def handle_screen_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """
        Handle /screen command.

        Usage:
          /screen list
          /screen oversold
          /screen value --sector tech
          /screen momentum --limit 5
        """
        if not context.args:
            await self._send_screen_help(update)
            return

        screen_name = context.args[0].lower()

        # Handle special commands
        if screen_name == "list":
            await self._send_screen_list(update)
            return

        if screen_name == "help":
            await self._send_screen_help(update)
            return

        # Parse options
        options = self._parse_options(context.args[1:])
        sector = options.get("sector")
        limit = int(options.get("limit", 10))
        universe = options.get("universe", "sp100")

        # Send initial message
        progress_msg = await update.message.reply_text(
            f"🔍 Running '{screen_name}' screen...\nThis may take 1-2 minutes."
        )

        try:
            # Run screen in background thread
            results = await asyncio.to_thread(
                self.screens.run_screen,
                screen_name,
                universe=universe,
                sector=sector,
                limit=limit
            )

            if not results:
                await progress_msg.edit_text(
                    f"❌ No stocks found matching '{screen_name}' criteria."
                )
                return

            # Format results
            message = self._format_screen_results(screen_name, results, sector, universe)

            await progress_msg.delete()
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except ValueError as e:
            await progress_msg.edit_text(f"❌ Error: {str(e)}")
        except Exception as e:
            await progress_msg.edit_text(f"❌ Screening failed: {str(e)}")

    async def handle_screener_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """
        Handle /screener command - tiered batch analysis.

        Usage:
          /screener run
          /screener run --sector tech
        """
        if not context.args or context.args[0].lower() != "run":
            await update.message.reply_text(
                "🔍 *Screener - Tiered Analysis*\n\n"
                "Run comprehensive tiered screening:\n"
                "• Quick filter on full universe\n"
                "• Medium analysis on top candidates\n"
                "• Deep analysis on best prospects\n\n"
                "*Usage:*\n"
                "`/screener run` - Screen all S&P 100\n"
                "`/screener run --sector tech` - Screen tech sector only\n\n"
                "⚠️ This takes 10-15 minutes for full universe.",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        # Parse options
        options = self._parse_options(context.args[1:])
        sector = options.get("sector")
        universe_name = options.get("universe", "sp100")

        # Get tickers
        if sector:
            tickers = self.universe.get_tickers_by_sector(sector, universe_name)
            if not tickers:
                await update.message.reply_text(f"❌ Sector '{sector}' not found.")
                return
        else:
            tickers = self.universe.get_tickers(universe_name)

        # Send initial message
        await update.message.reply_text(
            f"🚀 Starting tiered screening on {len(tickers)} stocks...\n"
            f"Universe: {universe_name.upper()}\n"
            f"Sector: {sector or 'All'}\n\n"
            f"This will take approximately 10-15 minutes.\n"
            f"I'll send progress updates as we go."
        )

        # Create a progress tracker
        async def send_progress(msg: str):
            await update.message.reply_text(msg)

        try:
            # Run tiered screening (in background)
            await update.message.reply_text("🔍 **Tier 1:** Quick analysis starting...")

            results = await asyncio.to_thread(
                self.analyzer.tiered_screening,
                tickers,
                quick_top_n=20,
                medium_top_n=10,
                deep_top_n=5
            )

            # Send final results
            message = self._format_tiered_results(results)
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            await update.message.reply_text(f"❌ Screening failed: {str(e)}")

    def _format_screen_results(
        self,
        screen_name: str,
        results,
        sector: Optional[str],
        universe: str
    ) -> str:
        """
        Format screening results for Telegram.

        Args:
            screen_name: Name of the screen
            results: List of ScreenResult objects
            sector: Optional sector filter
            universe: Universe name

        Returns:
            Formatted markdown message
        """
        # Get screen description
        descriptions = self.screens.list_screens()
        description = descriptions.get(screen_name, "")

        message = f"🔍 *{screen_name.upper()} Screen Results*\n\n"
        message += f"_{description}_\n\n"
        message += f"Universe: {universe.upper()}"
        if sector:
            message += f" | Sector: {sector.title()}"
        message += f"\nFound: *{len(results)} matches*\n\n"
        message += "─" * 30 + "\n\n"

        # Show top results
        for i, result in enumerate(results[:10], 1):
            metrics = result.metrics
            price = metrics.get("price", 0)
            rsi = metrics.get("rsi", 0)
            pe = metrics.get("pe_ratio", "N/A")
            returns_5d = metrics.get("returns_5d", 0)

            message += f"*{i}. {result.ticker}* - Score: {result.score:.0f}/100\n"
            message += f"   💰 ${price:.2f} | RSI: {rsi:.0f} | P/E: {pe}"
            if returns_5d != 0:
                message += f" | 5d: {returns_5d:+.1f}%"
            message += "\n\n"

        message += "─" * 30 + "\n"
        message += f"💡 Use `/analyze TICKER` for full report\n"
        message += f"📊 Use `/screen list` to see all screens"

        return message

    def _format_tiered_results(self, results: dict) -> str:
        """
        Format tiered screening results.

        Args:
            results: Dictionary with quick, medium, deep results

        Returns:
            Formatted markdown message
        """
        summary = results["summary"]
        deep_results = results["deep"]

        message = "🎯 *Tiered Screening Complete*\n\n"
        message += f"📊 *Summary:*\n"
        message += f"• Screened: {summary['total_screened']} stocks\n"
        message += f"• Quick passed: {summary['quick_passed']}\n"
        message += f"• Medium passed: {summary['medium_passed']}\n"
        message += f"• Deep analyzed: {summary['deep_analyzed']}\n\n"
        message += "─" * 30 + "\n\n"

        if deep_results:
            message += "*🏆 Top Recommendations:*\n\n"

            for i, result in enumerate(deep_results[:5], 1):
                message += f"*{i}. {result.ticker}*\n"
                message += f"   Score: {result.score:.0f}/100\n"
                message += f"   Signal: {result.signal.upper()}\n"
                message += f"   Time: {result.execution_time:.0f}s\n\n"

            message += "─" * 30 + "\n"
            message += "💡 Use `/analyze TICKER` for detailed analysis"
        else:
            message += "❌ No strong candidates found in deep analysis."

        return message

    async def _send_screen_list(self, update: Update):
        """Send list of available screens."""
        screens = self.screens.list_screens()

        message = "📋 *Available Screens*\n\n"

        # Group by category
        technical = ["oversold", "overbought", "breakout", "momentum", "reversal"]
        fundamental = ["value", "growth", "quality", "dividend"]
        hybrid = ["contrarian"]

        message += "*Technical Screens:*\n"
        for name in technical:
            if name in screens:
                message += f"• `{name}` - {screens[name]}\n"

        message += "\n*Fundamental Screens:*\n"
        for name in fundamental:
            if name in screens:
                message += f"• `{name}` - {screens[name]}\n"

        message += "\n*Hybrid Screens:*\n"
        for name in hybrid:
            if name in screens:
                message += f"• `{name}` - {screens[name]}\n"

        message += "\n─────────────\n"
        message += "*Usage:*\n"
        message += "`/screen oversold`\n"
        message += "`/screen value --sector tech`\n"
        message += "`/screen momentum --limit 5`"

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def _send_screen_help(self, update: Update):
        """Send screening help message."""
        message = """🔍 *Stock Screener Help*

*Basic Usage:*
`/screen <screen_name>`

*Examples:*
`/screen oversold` - Find oversold stocks
`/screen value` - Find undervalued stocks
`/screen momentum --limit 5` - Top 5 momentum stocks

*Options:*
`--sector <name>` - Filter by sector (tech, healthcare, etc.)
`--limit <number>` - Maximum results (default: 10)
`--universe <name>` - Stock universe (default: sp100)

*Available Screens:*
Use `/screen list` to see all available screens

*Advanced:*
`/screener run` - Tiered analysis (10-15 min)
  • Quick filter → Medium analysis → Deep analysis
  • Best for finding top opportunities

*Sectors:*
Technology, Healthcare, Financials, Energy,
Consumer Discretionary, Consumer Staples,
Industrials, Communication Services, Utilities

─────────────
💡 *Tips:*
• Start with a pre-built screen
• Use sector filters to narrow results
• Full analysis available via `/analyze TICKER`
"""
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    def _parse_options(self, args: list) -> dict:
        """
        Parse command line options.

        Args:
            args: List of arguments

        Returns:
            Dictionary of parsed options
        """
        options = {}
        i = 0

        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                key = arg[2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    options[key] = args[i + 1]
                    i += 2
                else:
                    options[key] = True
                    i += 1
            else:
                i += 1

        return options
