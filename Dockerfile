# Multi-stage build for StockSquad
FROM node:20-alpine AS frontend-builder

WORKDIR /build
COPY ui/web/package*.json ./
RUN ls
RUN npm ci
COPY ui/web/ ./
RUN npm run build

# Python runtime
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from builder stage
COPY --from=frontend-builder /build/dist ./ui/web/dist

# Create ChromaDB directory (using /tmp for local ephemeral storage)
RUN mkdir -p /tmp/chromadb && chmod 755 /tmp/chromadb

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMA_DB_PATH=/tmp/chromadb

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/reports || exit 1

# Start script
COPY docker-startup.sh /app/docker-startup.sh
RUN chmod +x /app/docker-startup.sh

CMD ["/app/docker-startup.sh"]
