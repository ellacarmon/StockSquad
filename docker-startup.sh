#!/bin/bash
# Docker container startup script for StockSquad

set -e

echo "========================================="
echo "StockSquad Container Starting"
echo "========================================="
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "ChromaDB path: ${CHROMA_DB_PATH:-/tmp/chromadb} (ephemeral)"
echo "========================================="

# Telegram bot disabled in container deployment
# Use the API endpoints instead for stock analysis

# Start FastAPI server with gunicorn
# Note: Using 1 worker because ChromaDB uses SQLite which doesn't support concurrent access
echo "Starting FastAPI server on 0.0.0.0:8000 with 1 worker..."
exec gunicorn ui.api:app \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
