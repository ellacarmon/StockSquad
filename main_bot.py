#!/usr/bin/env python3
"""
StockSquad Telegram Bot Entry Point

Run this to start the Telegram bot interface.
"""

from telegram_bot.bot import main

if __name__ == "__main__":
    print("=" * 60)
    print("StockSquad Telegram Bot")
    print("=" * 60)
    print("\nStarting bot...")
    print("Send /start in Telegram to begin")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 60)

    main()
