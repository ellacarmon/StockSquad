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
echo "Python version: $(python --version)"
echo "Node version: $(node --version 2>/dev/null || echo 'Node not found')"
echo "Working directory: $(pwd)"
echo "ChromaDB path: ${CHROMA_DB_PATH:-/mnt/chromadb}"

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Build frontend if not already built
if [ ! -d "ui/web/dist" ]; then
    echo "Building React frontend..."
    if [ -d "ui/web" ]; then
        cd ui/web
        npm install
        npm run build
        cd ../..
    else
        echo "Warning: ui/web directory not found, skipping frontend build"
    fi
else
    echo "Frontend already built, skipping..."
fi

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
