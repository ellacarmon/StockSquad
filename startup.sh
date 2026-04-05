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
echo "Python version: $(python3 --version 2>/dev/null || echo 'Python not found')"
echo "Node version: $(node --version 2>/dev/null || echo 'Node not found')"
echo "Working directory: $(pwd)"
echo "ChromaDB path: ${CHROMA_DB_PATH:-/mnt/chromadb}"

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Frontend is built in GitHub Actions, so we skip it here
echo "Frontend build handled by GitHub Actions, skipping..."

# Start Telegram bot in background (if token is configured)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Starting Telegram bot in background..."
    python3 main_bot.py &
    BOT_PID=$!
    echo "Telegram bot started with PID: $BOT_PID"
else
    echo "TELEGRAM_BOT_TOKEN not set, skipping Telegram bot"
fi

# Start FastAPI server
echo "Starting FastAPI server..."
echo "Listening on 0.0.0.0:8000"
echo "========================================="

# Use gunicorn for production (better than uvicorn for App Service)
pip install gunicorn uvicorn[standard]

# Start with gunicorn
exec gunicorn ui.api:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
