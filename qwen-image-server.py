from modelscope import DiffusionPipeline
import torch
import time
from datetime import datetime
from PIL import ImageDraw, ImageFont
import os
from flask import Flask, request, jsonify, send_file
import io
import threading
from functools import wraps

app = Flask(__name__)

# Global variables for model and metadata
pipe = None
model_load_time = 0
total_params = 0
total_size_mb = 0
device = None
torch_dtype = None
gpu_name = None
cuda_version = None
generation_lock = threading.Lock()  # Lock for thread-safe GPU access
request_counter = 0
counter_lock = threading.Lock()

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

aspect_ratios = {
    "s1:1": (512, 512),
    "m1:1": (512, 512),
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

def load_model():
    """Load the model once at startup"""
    global pipe, model_load_time, total_params, total_size_mb, device, torch_dtype, gpu_name, cuda_version
    
    model_name = "Qwen/Qwen-Image"
    
    # Determine device and dtype
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    # Record model loading time
    print("="*60)
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Data type: {torch_dtype}")
    print("="*60)
    
    model_load_start = time.time()
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    model_load_end = time.time()
    model_load_time = model_load_end - model_load_start
    
    # Calculate model size
    total_params = 0
    total_size_mb = 0
    try:
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            for param in pipe.unet.parameters():
                total_params += param.numel()
                total_size_mb += param.numel() * param.element_size() / (1024 * 1024)
        else:
            for component_name in pipe.components:
                component = getattr(pipe, component_name, None)
                if component is not None and hasattr(component, 'parameters'):
                    for param in component.parameters():
                        total_params += param.numel()
                        total_size_mb += param.numel() * param.element_size() / (1024 * 1024)
    except Exception as e:
        print(f"Warning: Could not calculate model size: {e}")
        total_params = 0
        total_size_mb = 0
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    
    print("="*60)
    print(f"✓ Model loaded successfully!")
    print(f"  - Loading time: {model_load_time:.2f} seconds")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_size_mb:.2f} MB")
    print(f"  - GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB" if torch.cuda.is_available() else "  - Running on CPU")
    print("="*60)

def add_info_to_image(image, inference_time, width, height):
    """Add system info overlay to the image"""
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_memory = f"{torch.cuda.memory_allocated()/1024**2:.0f} MB" if torch.cuda.is_available() else "N/A"
    
    draw = ImageDraw.Draw(image)
    
    info_lines = [
        f"Qwen-Image",
        f"aspect ratio: {width}x{height}",
        f"Inference Time: {inference_time:.2f}s",
        f"GPU: {gpu_name}",
        f"CUDA: {cuda_version}",
        f"DateTime: {current_datetime}"
    ]
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Calculate dimensions
    max_text_width = 0
    total_text_height = 0
    line_heights = []
    
    for line in info_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_text_width = max(max_text_width, line_width)
        line_heights.append(line_height)
        total_text_height += line_height
    
    line_spacing = 5
    total_text_height += line_spacing * (len(info_lines) - 1)
    
    padding = 10
    rect_padding = 8
    
    # Draw background rectangle
    rect_x1 = width - max_text_width - padding - rect_padding * 2
    rect_y1 = padding - rect_padding
    rect_x2 = width - padding + rect_padding
    rect_y2 = padding + total_text_height + rect_padding
    
    draw.rectangle(
        [(rect_x1, rect_y1), (rect_x2, rect_y2)],
        fill=(0, 0, 0, 200)
    )
    
    # Draw text
    y_offset = padding
    for i, line in enumerate(info_lines):
        text_position = (width - max_text_width - padding - rect_padding, y_offset)
        draw.text(text_position, line, fill="white", font=font)
        y_offset += line_heights[i] + line_spacing
    
    return image

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": device,
        "gpu": gpu_name,
        "cuda_version": cuda_version
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image endpoint with thread-safe GPU access"""
    global request_counter
    
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Increment request counter
    with counter_lock:
        request_counter += 1
        current_request = request_counter
    
    try:
        # Parse request
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', ' ')
        aspect_ratio = data.get('aspect_ratio', 's1:1')
        num_inference_steps = data.get('num_inference_steps', 20)
        true_cfg_scale = data.get('true_cfg_scale', 4.0)
        seed = data.get('seed', 42)
        language = data.get('language', 'en')
        save_to_file = data.get('save_to_file', False)
        add_overlay = data.get('add_overlay', True)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Get dimensions
        if aspect_ratio not in aspect_ratios:
            return jsonify({"error": f"Invalid aspect ratio. Choose from: {list(aspect_ratios.keys())}"}), 400
        
        width, height = aspect_ratios[aspect_ratio]
        
        print(f"\n[Request #{current_request}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating image...")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Aspect ratio: {aspect_ratio} ({width}x{height})")
        
        # Use lock to ensure thread-safe GPU access
        # Only one generation at a time to avoid GPU memory conflicts
        with generation_lock:
            start_time = time.time()
            generator = torch.Generator(device=device).manual_seed(seed) if device == "cuda" else torch.Generator().manual_seed(seed)
            
            image = pipe(
                prompt=prompt + positive_magic.get(language, positive_magic["en"]),
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator
            ).images[0]
            
            end_time = time.time()
            inference_time = end_time - start_time
        
        # Add overlay if requested (outside lock - CPU operation)
        if add_overlay:
            image = add_info_to_image(image, inference_time, width, height)
        
        print(f"  ✓ [Request #{current_request}] Image generated in {inference_time:.2f}s")
        
        # Save to file if requested
        if save_to_file:
            tmp_dir = "tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(tmp_dir, f"Qwen-Image-{timestamp}-req{current_request}.png")
            image.save(output_path)
            print(f"  Image saved to: {output_path}")
            
            return jsonify({
                "status": "success",
                "request_id": current_request,
                "inference_time": round(inference_time, 2),
                "file_path": output_path,
                "width": width,
                "height": height
            })
        else:
            # Return image as bytes
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')
            
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "Qwen/Qwen-Image",
        "device": device,
        "dtype": str(torch_dtype),
        "gpu": gpu_name,
        "cuda_version": cuda_version,
        "model_load_time": round(model_load_time, 2),
        "total_parameters": total_params,
        "model_size_mb": round(total_size_mb, 2),
        "gpu_memory_mb": round(torch.cuda.memory_allocated()/1024**2, 2) if torch.cuda.is_available() else 0,
        "supported_aspect_ratios": list(aspect_ratios.keys()),
        "total_requests_processed": request_counter,
        "concurrent_support": "enabled"
    })

# Load model at module import time (works with both Flask dev server and Gunicorn)
print("\n" + "="*60)
print("Initializing Qwen Image Generation Server...")
print("="*60)
load_model()
print("\n" + "="*60)
print("✓ Model loaded and ready to serve requests")
print("="*60 + "\n")

if __name__ == '__main__':
    # Start Flask development server
    # Note: Model is already loaded at module import time
    print("\n" + "="*60)
    print("🚀 Starting Flask development server...")
    print("For production, use: gunicorn --config gunicorn_config.py qwen-image-server:app")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
