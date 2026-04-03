"""
Telegram Report Formatter
Formats analysis reports for Telegram with proper markdown.
"""

from typing import Dict, Any
import re


class TelegramFormatter:
    """Formats StockSquad reports for Telegram."""

    MAX_MESSAGE_LENGTH = 4096  # Telegram limit

    @staticmethod
    def escape_markdown(text: str) -> str:
        """
        Escape special characters for Telegram MarkdownV2.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        # Characters that need escaping in MarkdownV2
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    @staticmethod
    def format_progress_message(ticker: str, step: str) -> str:
        """
        Format a progress update message.

        Args:
            ticker: Stock ticker
            step: Current step description

        Returns:
            Formatted message
        """
        emoji_map = {
            "data": "📊",
            "technical": "📈",
            "sentiment": "📰",
            "social": "💬",
            "fundamental": "💰",
            "devil": "😈",
            "synthesis": "🔮"
        }

        emoji = "⚙️"
        for key, value in emoji_map.items():
            if key in step.lower():
                emoji = value
                break

        return f"{emoji} {step}..."

    @staticmethod
    def format_analysis_result(result: Dict[str, Any]) -> str:
        """
        Format complete analysis result for Telegram (CONDENSED VERSION).

        Args:
            result: Analysis result dictionary

        Returns:
            Formatted markdown string
        """
        ticker = result.get("ticker", "Unknown")
        final_report = result.get("final_report", "No report generated")
        exec_time = result.get("execution_time_seconds", 0)

        # Extract key sections only
        exec_summary = TelegramFormatter._extract_section(final_report, "Executive Summary")
        recommendation = TelegramFormatter._extract_section(final_report, "Final Recommendation")

        # Get technical signal from individual agent results if available
        technical_signal = "N/A"
        sentiment_score = "N/A"

        agent_results = result.get("agent_results", {})
        if "technical" in agent_results:
            tech = agent_results["technical"]
            if isinstance(tech, dict):
                technical_signal = tech.get("signal", {}).get("direction", "N/A")

        if "sentiment" in agent_results:
            sent = agent_results["sentiment"]
            if isinstance(sent, dict):
                sentiment_score = sent.get("overall_sentiment", "N/A")

        # Build condensed report
        message = f"📊 *{ticker} Analysis*\n\n"

        # Quick stats
        message += f"📈 Signal: *{technical_signal}*\n"
        message += f"📰 Sentiment: *{sentiment_score}*\n"
        message += f"⏱ Time: {exec_time:.1f}s\n\n"
        message += "─" * 30 + "\n\n"

        # Executive Summary (first 500 chars)
        if exec_summary:
            message += f"📋 *Executive Summary*\n{exec_summary[:500]}"
            if len(exec_summary) > 500:
                message += "..."
            message += "\n\n"

        # Final Recommendation (first 400 chars)
        if recommendation:
            message += f"🎯 *Recommendation*\n{recommendation[:400]}"
            if len(recommendation) > 400:
                message += "..."
            message += "\n\n"

        # Footer
        message += f"{'─' * 30}\n"
        message += f"💾 Saved • Use /history {ticker} for past analyses"

        return message

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """
        Extract a specific section from the report.

        Args:
            text: Full report text
            section_name: Name of section to extract

        Returns:
            Section content or empty string
        """
        # Try to find section with various markdown formats
        patterns = [
            rf'\*\*{section_name}\*\*:?\s*\n+(.*?)(?=\n\*\*|\n#|$)',
            rf'#{1,3}\s*{section_name}:?\s*\n+(.*?)(?=\n#|\n\*\*|$)',
            rf'{section_name}:?\s*\n+(.*?)(?=\n\n[A-Z]|\n#|\n\*\*|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Clean up markdown
                content = re.sub(r'\*\*', '', content)
                content = re.sub(r'\n+', ' ', content)
                return content

        return ""

    @staticmethod
    def split_message(message: str) -> list:
        """
        Split long messages into chunks under Telegram's limit.

        Args:
            message: Full message

        Returns:
            List of message chunks
        """
        if len(message) <= TelegramFormatter.MAX_MESSAGE_LENGTH:
            return [message]

        chunks = []
        current_chunk = ""

        lines = message.split('\n')

        for line in lines:
            if len(current_chunk) + len(line) + 1 > TelegramFormatter.MAX_MESSAGE_LENGTH:
                chunks.append(current_chunk)
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def format_error_message(error: str) -> str:
        """
        Format an error message.

        Args:
            error: Error description

        Returns:
            Formatted error message
        """
        return f"❌ *Error*\n\n{error}\n\nPlease try again or use /help for assistance."

    @staticmethod
    def format_help_message() -> str:
        """
        Format help message with available commands.

        Returns:
            Help message
        """
        return """🤖 *StockSquad Bot Commands*

*Analysis Commands:*
/analyze AAPL \\- Full multi\\-agent analysis
/screen oversold \\- Find stocks by criteria
/screener run \\- Tiered screening \\(10\\-15 min\\)

*Screening:*
/screen list \\- Show all available screens
/screen value \\-\\-sector tech \\- Sector filtering
/screen momentum \\-\\-limit 5 \\- Custom limits

*History Commands:*
/history TSLA \\- View past analyses
/stats \\- Memory statistics

*Help:*
/help \\- Show this message
/start \\- Welcome message

*Quick Usage:*
Just send any ticker \\(e\\.g\\., AAPL\\) and I'll analyze it\\!

─────────────
💡 *Tips:*
• Use /screen for quick stock discovery
• Use /analyze for detailed reports
• Full screening with /screener run
• All reports saved to memory

🚀 Try: /screen oversold"""

    @staticmethod
    def format_welcome_message() -> str:
        """
        Format welcome message.

        Returns:
            Welcome message
        """
        return """👋 *Welcome to StockSquad\\!*

I'm your AI\\-powered stock research assistant with a team of specialized agents:

🤖 *The Squad:*
• 📊 DataAgent \\- Market data
• 📈 TechnicalAgent \\- Charts \\& signals
• 📰 SentimentAgent \\- News analysis
• 💬 SocialMediaAgent \\- X \\& Reddit
• 💰 FundamentalsAgent \\- Financials
• 😈 DevilsAdvocate \\- Risk challenges

*How to use:*
Just send me a ticker \\(e\\.g\\., AAPL\\) or use /analyze AAPL

Commands: /help
"""

    @staticmethod
    def format_history_message(ticker: str, analyses: list) -> str:
        """
        Format historical analyses for a ticker.

        Args:
            ticker: Stock ticker
            analyses: List of past analyses

        Returns:
            Formatted message
        """
        if not analyses:
            return f"No past analyses found for {ticker}"

        message = f"📜 *Analysis History: {ticker}*\n\n"

        for i, analysis in enumerate(analyses[:5], 1):  # Show last 5
            timestamp = analysis.get("timestamp", "Unknown")[:10]
            summary = analysis.get("summary", "No summary")[:100]

            message += f"{i}\\. *{timestamp}*\n"
            message += f"   {summary}\\.\\.\\.\n\n"

        message += f"\n📊 Total analyses: {len(analyses)}"

        return message
