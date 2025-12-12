"""
Gunicorn configuration file for Qwen Image Generation Server

This configuration enables concurrent request handling:
- Workers: Number of worker processes
- Threads: Number of threads per worker
- Worker class: gevent for async I/O handling
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
# Note: For GPU workloads, typically use 1-2 workers to avoid GPU memory conflicts
# Each worker loads the model into GPU memory
workers = int(os.getenv('WORKERS', '1'))  # Default to 1 worker for GPU

# Worker class
# Use 'gevent' for async I/O or 'sync' for standard synchronous workers
worker_class = 'gevent'

# Number of concurrent requests per worker
# With gevent, this allows handling multiple requests concurrently
worker_connections = 1000
threads = int(os.getenv('THREADS', '4'))  # Threads per worker

# Worker timeout
timeout = 300  # 5 minutes - image generation can take time
keepalive = 5

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'qwen-image-server'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Preload app for faster worker spawn (careful with GPU memory)
preload_app = False  # Set to False for GPU workloads to avoid memory issues
# Each worker will load the model independently when the module is imported

# Worker lifecycle
max_requests = 1000  # Restart workers after N requests to prevent memory leaks
max_requests_jitter = 50  # Add randomness to prevent all workers restarting at once

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

def on_starting(server):
    """Called just before the master process is initialized."""
    print("="*60)
    print("🚀 Starting Qwen Image Generation Server")
    print(f"   Workers: {workers}")
    print(f"   Threads per worker: {threads}")
    print(f"   Worker class: {worker_class}")
    print(f"   Bind: {bind}")
    print("="*60)

def when_ready(server):
    """Called just after the server is started."""
    print("✓ Server is ready to accept connections")

def on_exit(server):
    """Called just before exiting."""
    print("Shutting down Qwen Image Generation Server...")
