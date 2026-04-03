"""
Telegram Bot for StockSquad
Provides chat-based interface for stock analysis.
"""

import os
import logging
from typing import Optional
import asyncio

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes
    )
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Warning: python-telegram-bot not installed")
    print("Install with: pip install python-telegram-bot")

from agents.orchestrator import OrchestratorAgent
from telegram_bot.formatter import TelegramFormatter
from telegram_bot.screener_handler import ScreenerHandler
from config import get_settings


# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class StockSquadBot:
    """Telegram bot for StockSquad stock analysis."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the bot.

        Args:
            token: Telegram bot token (or from environment)
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot not installed")

        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")

        self.formatter = TelegramFormatter()
        self.orchestrator = OrchestratorAgent()
        self.screener_handler = ScreenerHandler()

        # Optional: Chat ID whitelist for security
        allowed_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.allowed_chat_id = int(allowed_chat_id) if allowed_chat_id.strip() else None

        self.application = None

    def is_chat_allowed(self, chat_id: int) -> bool:
        """
        Check if chat is allowed to use the bot.

        Args:
            chat_id: Telegram chat ID

        Returns:
            True if allowed
        """
        if self.allowed_chat_id is None:
            return True  # No whitelist = everyone allowed
        return chat_id == self.allowed_chat_id

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        message = self.formatter.format_welcome_message()
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            return

        message = self.formatter.format_help_message()
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze TICKER command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        # Get ticker from command args
        if not context.args:
            await update.message.reply_text(
                "Please provide a ticker.\nUsage: /analyze AAPL"
            )
            return

        ticker = context.args[0].upper()

        # Send initial message
        progress_msg = await update.message.reply_text(
            f"🔍 Analyzing {ticker}...\nThis will take 60-90 seconds."
        )

        try:
            # Run analysis (this is blocking, but that's okay for now)
            # We send progress updates
            await progress_msg.edit_text(
                self.formatter.format_progress_message(ticker, "Collecting data")
            )

            # Run the analysis
            result = await asyncio.to_thread(
                self.orchestrator.analyze_stock,
                ticker
            )

            if not result.get("success", False):
                error_msg = self.formatter.format_error_message(
                    result.get("error", "Analysis failed")
                )
                await progress_msg.edit_text(error_msg, parse_mode=ParseMode.MARKDOWN)
                return

            # Format and send result
            formatted_result = self.formatter.format_analysis_result(result)

            # Split if too long
            messages = self.formatter.split_message(formatted_result)

            # Delete progress message
            await progress_msg.delete()

            # Send result(s)
            for i, msg in enumerate(messages):
                if i == 0:
                    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
                else:
                    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            error_msg = self.formatter.format_error_message(str(e))
            await progress_msg.edit_text(error_msg, parse_mode=ParseMode.MARKDOWN)

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history TICKER command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            return

        if not context.args:
            await update.message.reply_text(
                "Please provide a ticker.\nUsage: /history AAPL"
            )
            return

        ticker = context.args[0].upper()

        try:
            analyses = self.orchestrator.get_past_analyses(ticker, limit=5)
            message = self.formatter.format_history_message(ticker, analyses)
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

        except Exception as e:
            logger.error(f"History error: {e}")
            await update.message.reply_text(f"❌ Error retrieving history: {str(e)}")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            return

        try:
            from memory.long_term import LongTermMemory
            memory = LongTermMemory()
            stats = memory.get_collection_stats()

            message = f"""📊 *StockSquad Statistics*

Total Analyses: {stats.get('total_analyses', 0)}
Storage: {stats.get('storage_path', 'N/A')}

Use /history TICKER to view past analyses"""

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Stats error: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def screen_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /screen command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            return

        await self.screener_handler.handle_screen_command(update, context)

    async def screener_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /screener command."""
        if not self.is_chat_allowed(update.effective_chat.id):
            return

        await self.screener_handler.handle_screener_command(update, context)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle plain text messages (treat as ticker)."""
        if not self.is_chat_allowed(update.effective_chat.id):
            return

        text = update.message.text.strip().upper()

        # Check if it looks like a ticker (2-5 capital letters)
        if len(text) >= 2 and len(text) <= 5 and text.isalpha():
            # Treat as ticker and analyze
            context.args = [text]
            await self.analyze_command(update, context)
        else:
            await update.message.reply_text(
                "Send me a stock ticker (e.g., AAPL) or use /help for commands."
            )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        logger.error(f"Update {update} caused error {context.error}")

        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again or contact support."
            )

    def run(self):
        """Run the bot."""
        if not self.token:
            raise ValueError("Bot token not configured")

        logger.info("Starting StockSquad Telegram Bot...")

        # Build application
        self.application = Application.builder().token(self.token).build()

        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("history", self.history_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("screen", self.screen_command))
        self.application.add_handler(CommandHandler("screener", self.screener_command))

        # Handle plain text messages (tickers)
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        # Error handler
        self.application.add_error_handler(self.error_handler)

        # Run bot
        logger.info("Bot started! Send /start to begin.")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point for bot."""
    try:
        bot = StockSquadBot()
        bot.run()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set TELEGRAM_BOT_TOKEN in .env file")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Install dependencies: pip install python-telegram-bot")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
