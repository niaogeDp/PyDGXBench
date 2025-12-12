from modelscope import DiffusionPipeline
import torch
import time
from datetime import datetime
from PIL import ImageDraw, ImageFont

model_name = "Qwen/Qwen-Image"

# Load the pipeline
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

print("="*60)
print(f"✓ Model loaded successfully!")
print(f"  - Loading time: {model_load_time:.2f} seconds")

# Calculate model size
total_params = 0
total_size_mb = 0
try:
    # Try to get parameters from unet
    if hasattr(pipe, 'unet') and pipe.unet is not None:
        for param in pipe.unet.parameters():
            total_params += param.numel()
            total_size_mb += param.numel() * param.element_size() / (1024 * 1024)
    else:
        # Fallback: iterate through all components
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


print(f"  - Total parameters: {total_params:,}")
print(f"  - Model size: {total_size_mb:.2f} MB")
print(f"  - GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB" if torch.cuda.is_available() else "  - Running on CPU")
print("="*60)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = 'A movie poster. The first row is the movie title, which reads “Imagination Unleashed”. The second row is the movie subtitle, which reads “Enter a world beyond your imagination”. The third row reads “Cast: Qwen-Image”. The fourth row reads “Director: The Collective Imagination of Humanity”. The central visual features a sleek, futuristic computer from which radiant colors, whimsical creatures, and dynamic, swirling patterns explosively emerge, filling the composition with energy, motion, and surreal creativity. The background transitions from dark, cosmic tones into a luminous, dreamlike expanse, evoking a digital fantasy realm. At the bottom edge, the text “Launching in the Cloud, August 2025” appears in bold, modern sans-serif font with a glowing, slightly transparent effect, evoking a high-tech, cinematic aesthetic. The overall style blends sci-fi surrealism with graphic design flair—sharp contrasts, vivid color grading, and layered visual depth—reminiscent of visionary concept art and digital matte painting,'
negative_prompt = " " # using an empty string if you do not have specific concept to remove


# Generate with different aspect ratios
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

# width, height = aspect_ratios["16:9"]
width, height = aspect_ratios["s1:1"]

# Record inference time
start_time = time.time()
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=20,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]
end_time = time.time()
inference_time = end_time - start_time

# Collect system information
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
gpu_memory = f"{torch.cuda.memory_allocated()/1024**2:.0f} MB" if torch.cuda.is_available() else "N/A"

# Add inference time and system info to the top-right corner of the image
draw = ImageDraw.Draw(image)

# Create multi-line text with all information
info_lines = [
    f"Load Time: {model_load_time:.2f}s",
    f"Inference Time: {inference_time:.2f}s",
    f"Model Size: {total_size_mb:.0f} MB",
    # f"Parameters: {total_params:,}",
    f"GPU: {gpu_name}",
    f"GPU Memory: {gpu_memory}",
    f"CUDA: {cuda_version}",
    f"DateTime: {current_datetime}"
]

# Try to use a default font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

# Calculate the maximum width and total height needed
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

# Add spacing between lines
line_spacing = 5
total_text_height += line_spacing * (len(info_lines) - 1)

# Position at top-right corner with some padding
padding = 10
rect_padding = 8

# Draw background rectangle for better visibility
rect_x1 = width - max_text_width - padding - rect_padding * 2
rect_y1 = padding - rect_padding
rect_x2 = width - padding + rect_padding
rect_y2 = padding + total_text_height + rect_padding

draw.rectangle(
    [(rect_x1, rect_y1), (rect_x2, rect_y2)],
    fill=(0, 0, 0, 200)
)

# Draw each line of text in white
y_offset = padding
for i, line in enumerate(info_lines):
    text_position = (width - max_text_width - padding - rect_padding, y_offset)
    draw.text(text_position, line, fill="white", font=font)
    y_offset += line_heights[i] + line_spacing

print(f"Model loading time: {model_load_time:.2f} seconds")
print(f"Inference completed in {inference_time:.2f} seconds")
print(f"GPU: {gpu_name}")
print(f"CUDA Version: {cuda_version}")
print(f"DateTime: {current_datetime}")

from datetime import datetime
import os

# Create tmp directory if it doesn't exist
tmp_dir = "tmp"
os.makedirs(tmp_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(tmp_dir, f"Qwen-Image-{timestamp}.png")
image.save(output_path)
print(f"Image saved to: {output_path}")
