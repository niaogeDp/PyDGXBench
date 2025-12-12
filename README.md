# Qwen Image Generation Server

A production-ready Flask server for the Qwen-Image model with **concurrent request handling** support.

## 🌟 Features

- ✅ **Model preloading** - Loads model once at startup, avoiding repeated loading delays
- ✅ **Persistent service** - Keeps running and waits for requests without terminating
- ✅ **Concurrent request support** - Handles multiple requests with thread-safe GPU access
- ✅ **Thread-safe processing** - Uses locks to prevent GPU memory conflicts
- ✅ **Request tracking** - Tracks and logs all requests with unique IDs
- ✅ **Production-ready** - Uses Gunicorn with gevent for async I/O handling
- ✅ **Automatic info overlay** - Displays GPU, CUDA, timing info on generated images
- ✅ **Flexible output** - Return images as bytes or save to file

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Start Server (Windows)
```bash
start_server.bat
```

### Start Server (Linux/Mac)
```bash
bash start_server.sh
```

### Start with Custom Configuration
```bash
# Windows
set WORKERS=1
set THREADS=4
set PORT=5000
start_server.bat

# Linux/Mac
WORKERS=1 THREADS=4 PORT=5000 bash start_server.sh
```

## 📋 API Endpoints

### 1. Health Check
**GET** `/health`

Check if the server is running and healthy.

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4090",
  "cuda_version": "12.1"
}
```

### 2. Model Information
**GET** `/info`

Get detailed model and server information.

```bash
curl http://localhost:5000/info
```

Response:
```json
{
  "model_name": "Qwen/Qwen-Image",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4090",
  "cuda_version": "12.1",
  "model_load_time": 45.32,
  "total_parameters": 2500000000,
  "model_size_mb": 9536.74,
  "gpu_memory_mb": 9845,
  "supported_aspect_ratios": ["s1:1", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
  "total_requests_processed": 123,
  "concurrent_support": "enabled"
}
```

### 3. Generate Image
**POST** `/generate`

Generate an image based on the prompt.

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute cat sitting on a windowsill",
    "aspect_ratio": "16:9",
    "save_to_file": true
  }' \
  --output image.png
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Image description |
| `negative_prompt` | string | `" "` | Concepts to avoid |
| `aspect_ratio` | string | `"s1:1"` | Image size: `s1:1`, `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3` |
| `num_inference_steps` | int | `20` | Number of denoising steps |
| `true_cfg_scale` | float | `4.0` | Classifier-free guidance scale |
| `seed` | int | `42` | Random seed for reproducibility |
| `language` | string | `"en"` | Prompt language: `en` or `zh` |
| `save_to_file` | bool | `false` | Save to tmp folder (returns JSON) or return image bytes |
| `add_overlay` | bool | `true` | Add info overlay to image |

#### Response (save_to_file=true)
```json
{
  "status": "success",
  "request_id": 5,
  "inference_time": 3.45,
  "file_path": "tmp/Qwen-Image-20231212_153045-req5.png",
  "width": 1664,
  "height": 928
}
```

#### Response (save_to_file=false)
Returns image as PNG bytes (mimetype: `image/png`)

## 🧪 Testing

### Basic Client Test
```bash
python test_client.py
```

### Concurrent Request Test
```bash
python test_concurrent.py
```

This will test:
- Sequential requests (baseline)
- Concurrent requests (parallel)
- Performance comparison

## 🔧 Configuration

### Gunicorn Configuration (`gunicorn_config.py`)

Key settings:
- **workers**: Number of worker processes (default: 1 for GPU)
- **threads**: Threads per worker (default: 4)
- **worker_class**: `gevent` for async I/O
- **timeout**: 300 seconds (5 minutes)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | `1` | Number of worker processes |
| `THREADS` | `4` | Threads per worker |
| `PORT` | `5000` | Server port |

## 🎯 Concurrent Request Handling

The server supports concurrent request handling with the following architecture:

### Thread-Safe GPU Access
- Uses `threading.Lock()` to ensure thread-safe GPU access
- Multiple requests can be queued, but GPU operations are serialized
- Prevents GPU memory conflicts and CUDA errors

### Request Processing Flow
1. Multiple requests arrive concurrently
2. Requests are queued and assigned unique IDs
3. GPU operations are processed sequentially using locks
4. CPU operations (image overlay, file I/O) run in parallel
5. Responses are returned to clients independently

### Performance Notes

**For GPU Workloads:**
- Concurrent requests are **queued** but processed **sequentially** on GPU
- This ensures stability and prevents out-of-memory errors
- Wall-clock time for N concurrent requests ≈ N × single request time
- The benefit is handling multiple clients without server restarts

**For CPU Workloads:**
- True parallel processing is possible
- Can process multiple requests simultaneously
- Significant speedup with concurrent requests

## 📊 Example Usage

### Python Client
```python
import requests

# Generate image
response = requests.post('http://localhost:5000/generate', json={
    'prompt': 'A beautiful sunset over mountains',
    'aspect_ratio': '16:9',
    'num_inference_steps': 20,
    'seed': 42,
    'save_to_file': True
})

result = response.json()
print(f"Image saved to: {result['file_path']}")
print(f"Inference time: {result['inference_time']}s")
```

### cURL
```bash
# Save to file
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cute cat", "save_to_file": true}'

# Get image bytes
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cute cat", "save_to_file": false}' \
  --output image.png
```

## 🎨 Supported Aspect Ratios

| Name | Resolution | Use Case |
|------|------------|----------|
| `s1:1` | 512×512 | Small square (fast) |
| `m1:1` | 512×512 | Medium square |
| `1:1` | 1328×1328 | Large square |
| `16:9` | 1664×928 | Widescreen |
| `9:16` | 928×1664 | Portrait/Mobile |
| `4:3` | 1472×1140 | Standard |
| `3:4` | 1140×1472 | Portrait |
| `3:2` | 1584×1056 | Photo |
| `2:3` | 1056×1584 | Portrait Photo |

## 📝 Files

- `qwen-image-server.py` - Main Flask server with concurrent support
- `qwen-image-single.py` - Single-use script (original)
- `gunicorn_config.py` - Gunicorn production configuration
- `start_server.sh` - Linux/Mac startup script
- `start_server.bat` - Windows startup script
- `test_client.py` - Basic API testing client
- `test_concurrent.py` - Concurrent request testing
- `requirements.txt` - Python dependencies

## 🔍 Troubleshooting

### Server won't start
```bash
# Check if port is already in use
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/Mac

# Try a different port
PORT=5001 bash start_server.sh
```

### Out of GPU memory
```bash
# Reduce to 1 worker
WORKERS=1 bash start_server.sh

# Use smaller aspect ratios
curl ... -d '{"prompt": "...", "aspect_ratio": "s1:1"}'
```

### Slow response times
- Check GPU availability: Should use CUDA, not CPU
- Reduce `num_inference_steps` (default: 20, min: 10)
- Use smaller aspect ratios for faster generation

## 📚 Additional Resources

- [Qwen-Image Model Card](https://huggingface.co/Qwen/Qwen-Image)
- [ModelScope Documentation](https://modelscope.cn/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)

## 🙋 FAQ

**Q: Does the server support true parallel image generation?**  
A: The server can handle multiple concurrent requests, but GPU operations are serialized using thread-safe locks to prevent memory conflicts. This means requests are queued and processed one at a time on the GPU, but the server remains responsive to multiple clients.

**Q: How many workers should I use?**  
A: For GPU workloads, use 1-2 workers (default: 1). Each worker loads the full model into GPU memory. For CPU workloads, you can use more workers.

**Q: Can I increase processing speed?**  
A: Yes! Reduce `num_inference_steps` (20→10), use smaller aspect ratios (`s1:1`), or ensure you're using GPU instead of CPU.

**Q: How do I stop the server?**  
A: Press `Ctrl+C` in the terminal where the server is running.

---

**Built with ❤️ for Qwen-Image**
