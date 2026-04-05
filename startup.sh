#!/bin/bash
# Azure App Service startup script for StockSquad

set -e

echo "========================================="
echo "StockSquad Startup Script"
echo "========================================="

# Ensure ChromaDB directory exists
mkdir -p /mnt/chromadb
chmod 755 /mnt/chromadb

# Display environment info
echo "Python version: $(python --version 2>/dev/null || python3 --version 2>/dev/null || echo 'Python not found')"
echo "Working directory: $(pwd)"
echo "ChromaDB path: ${CHROMA_DB_PATH:-/mnt/chromadb}"

# Dependencies are installed by Oryx during deployment
# No need to install them here - this makes startup much faster

# Start Telegram bot in background (if token is configured)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Starting Telegram bot in background..."
    python main_bot.py &
    BOT_PID=$!
    echo "Telegram bot started with PID: $BOT_PID"
else
    echo "TELEGRAM_BOT_TOKEN not set, skipping Telegram bot"
fi

# Start FastAPI server
echo "Starting FastAPI server..."
echo "Listening on 0.0.0.0:8000"
echo "========================================="

# Start with gunicorn (installed via requirements.txt)
exec gunicorn ui.api:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
