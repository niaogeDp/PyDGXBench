#!/bin/bash
# Start Qwen Image Generation Server with Gunicorn
# This enables concurrent request handling

# Configuration
WORKERS=${WORKERS:-1}        # Number of worker processes (default: 1 for GPU)
THREADS=${THREADS:-4}        # Threads per worker (default: 4)
PORT=${PORT:-5000}           # Server port

echo "=========================================="
echo "Starting Qwen Image Server with Gunicorn"
echo "Workers: $WORKERS"
echo "Threads per worker: $THREADS"
echo "Port: $PORT"
echo "=========================================="

# Start server with gunicorn
gunicorn \
    --config gunicorn_config.py \
    --bind 0.0.0.0:$PORT \
    qwen-image-server:app

# Alternative: Run with custom worker/thread settings
# WORKERS=2 THREADS=4 gunicorn --config gunicorn_config.py qwen-image-server:app
