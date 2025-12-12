import requests
import json
from datetime import datetime

# Server URL
BASE_URL = "http://localhost:5000"

def check_health():
    """Check if the server is healthy"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def get_model_info():
    """Get model information"""
    response = requests.get(f"{BASE_URL}/info")
    print("Model Info:")
    print(json.dumps(response.json(), indent=2))
    print()

def generate_image_to_file():
    """Generate image and save to file"""
    payload = {
        "prompt": "A cute cat sitting on a colorful carpet",
        "negative_prompt": "blurry, low quality",
        "aspect_ratio": "s1:1",
        "num_inference_steps": 20,
        "true_cfg_scale": 4.0,
        "seed": 42,
        "language": "en",
        "save_to_file": True,
        "add_overlay": True
    }
    
    print("Generating image (save to file)...")
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    print()

def generate_image_return_bytes():
    """Generate image and return as bytes"""
    payload = {
        "prompt": "A beautiful sunset over the ocean with waves",
        "aspect_ratio": "16:9",
        "num_inference_steps": 20,
        "seed": 123,
        "save_to_file": False,
        "add_overlay": True
    }
    
    print("Generating image (return bytes)...")
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    
    if response.status_code == 200:
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client_image_{timestamp}.png"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Success! Image saved to: {filename}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    print()

def generate_chinese_prompt():
    """Generate image with Chinese prompt"""
    payload = {
        "prompt": "一只可爱的熊猫在竹林里吃竹子",
        "aspect_ratio": "1:1",
        "num_inference_steps": 20,
        "seed": 456,
        "language": "zh",
        "save_to_file": True,
        "add_overlay": True
    }
    
    print("Generating image with Chinese prompt...")
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    print()

if __name__ == "__main__":
    print("="*60)
    print("Qwen Image Generation Client Test")
    print("="*60)
    print()
    
    # Test 1: Health check
    check_health()
    
    # Test 2: Get model info
    get_model_info()
    
    # Test 3: Generate image and save to file
    generate_image_to_file()
    
    # Test 4: Generate image and return bytes
    generate_image_return_bytes()
    
    # Test 5: Generate with Chinese prompt
    generate_chinese_prompt()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)
