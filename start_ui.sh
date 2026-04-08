#!/bin/zsh
# Start FastAPI UI API with uvicorn (development mode)
uvicorn ui.api:app --reload --port 8000
