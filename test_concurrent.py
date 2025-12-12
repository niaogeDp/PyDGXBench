import requests
import json
import time
from datetime import datetime
import concurrent.futures
import threading

# Server URL
BASE_URL = "http://localhost:5000"

# Available aspect ratios
ASPECT_RATIOS = ["s1:1", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]

# Thread-safe counter for tracking results
results_lock = threading.Lock()
successful_requests = 0
failed_requests = 0
total_time = 0

def send_request(request_id, prompt, aspect_ratio="s1:1"):
    """Send a single image generation request"""
    global successful_requests, failed_requests, total_time
    
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "num_inference_steps": 20,
        "seed": request_id,  # Different seed for each request
        "save_to_file": True,
        "add_overlay": True
    }
    
    start_time = time.time()
    try:
        print(f"[Request {request_id}] Sending request (aspect_ratio={aspect_ratio}) at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=600)
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            with results_lock:
                successful_requests += 1
                total_time += elapsed
            print(f"[Request {request_id}] ✓ SUCCESS ({aspect_ratio}) in {elapsed:.2f}s - Inference: {result.get('inference_time', 0):.2f}s")
            return True, elapsed, result
        else:
            with results_lock:
                failed_requests += 1
            print(f"[Request {request_id}] ✗ FAILED: {response.status_code} - {response.text[:100]}")
            return False, elapsed, None
            
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        with results_lock:
            failed_requests += 1
        print(f"[Request {request_id}] ✗ ERROR in {elapsed:.2f}s: {str(e)[:100]}")
        return False, elapsed, None

def test_sequential(num_requests=3):
    """Test sequential requests (one after another)"""
    print("\n" + "="*60)
    print(f"Sequential Test: {num_requests} requests")
    print("="*60)
    
    prompts = [
        "A cute cat sitting on a windowsill",
        "A beautiful mountain landscape at sunset",
        "A futuristic city with flying cars",
        "A serene beach with palm trees",
        "A colorful hot air balloon in the sky"
    ]
    
    start_time = time.time()
    results = []
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        aspect_ratio = ASPECT_RATIOS[i % len(ASPECT_RATIOS)]  # Iterate through aspect ratios
        success, elapsed, result = send_request(i+1, prompt, aspect_ratio)
        results.append((success, elapsed))
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "-"*60)
    print(f"Sequential Test Results:")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Average time per request: {total_elapsed/num_requests:.2f}s")
    print(f"  Successful: {sum(1 for s, _ in results if s)}/{num_requests}")
    print("-"*60)
    
    return total_elapsed

def test_concurrent(num_requests=3, max_workers=3):
    """Test concurrent requests (parallel)"""
    global successful_requests, failed_requests, total_time
    successful_requests = 0
    failed_requests = 0
    total_time = 0
    
    print("\n" + "="*60)
    print(f"Concurrent Test: {num_requests} requests with {max_workers} workers")
    print("="*60)
    
    prompts = [
        "A cute cat sitting on a windowsill",
        "A beautiful mountain landscape at sunset",
        "A futuristic city with flying cars",
        "A serene beach with palm trees",
        "A colorful hot air balloon in the sky"
    ]
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            aspect_ratio = ASPECT_RATIOS[i % len(ASPECT_RATIOS)]  # Iterate through aspect ratios
            future = executor.submit(send_request, i+1, prompt, aspect_ratio)
            futures.append(future)
        
        # Wait for all to complete
        concurrent.futures.wait(futures)
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "-"*60)
    print(f"Concurrent Test Results:")
    print(f"  Total wall-clock time: {total_elapsed:.2f}s")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Successful: {successful_requests}/{num_requests}")
    print(f"  Failed: {failed_requests}/{num_requests}")
    if successful_requests > 0:
        print(f"  Average time per request: {total_time/successful_requests:.2f}s")
        print(f"  Speedup vs sequential: {total_time/total_elapsed:.2f}x")
    print("-"*60)
    
    return total_elapsed

def check_server_status():
    """Check if server is ready"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is healthy and ready")
            return True
    except:
        pass
    
    print("✗ Server is not responding. Please start the server first.")
    return False

def test_all_aspect_ratios():
    """Test all aspect ratios sequentially"""
    global successful_requests, failed_requests, total_time
    successful_requests = 0
    failed_requests = 0
    total_time = 0
    
    print("\n" + "="*60)
    print(f"Testing All Aspect Ratios ({len(ASPECT_RATIOS)} total)")
    print("="*60)
    
    prompt = "A beautiful landscape with mountains and a lake at sunset"
    start_time = time.time()
    results = []
    
    for i, aspect_ratio in enumerate(ASPECT_RATIOS):
        print(f"\nTesting {i+1}/{len(ASPECT_RATIOS)}: {aspect_ratio}")
        success, elapsed, result = send_request(i+1, prompt, aspect_ratio)
        results.append((aspect_ratio, success, elapsed, result))
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("Aspect Ratio Test Summary")
    print("="*60)
    print(f"{'Aspect Ratio':<15} {'Status':<10} {'Time (s)':<12} {'Inference (s)'}")
    print("-"*60)
    
    for aspect_ratio, success, elapsed, result in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        inference = f"{result.get('inference_time', 0):.2f}" if result else "N/A"
        print(f"{aspect_ratio:<15} {status:<10} {elapsed:>10.2f}   {inference:>10}")
    
    print("-"*60)
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average time: {total_time/successful_requests:.2f}s" if successful_requests > 0 else "Average time: N/A")
    print(f"Successful: {successful_requests}/{len(ASPECT_RATIOS)}")
    print(f"Failed: {failed_requests}/{len(ASPECT_RATIOS)}")
    print("="*60)
    
    return total_elapsed

if __name__ == "__main__":
    print("="*60)
    print("Qwen Image Server - Concurrent Request Test")
    print("="*60)
    
    # Check server status
    if not check_server_status():
        print("\nPlease start the server with:")
        print("  python start_server.bat  (Windows)")
        print("  or")
        print("  bash start_server.sh     (Linux/Mac)")
        exit(1)
    
    # Get server info
    try:
        response = requests.get(f"{BASE_URL}/info")
        if response.status_code == 200:
            info = response.json()
            print(f"\nServer Info:")
            print(f"  Device: {info.get('device')}")
            print(f"  GPU: {info.get('gpu')}")
            print(f"  Concurrent Support: {info.get('concurrent_support', 'unknown')}")
            print(f"  Total Requests Processed: {info.get('total_requests_processed', 0)}")
    except:
        pass
    
    # Run tests
    print("\n" + "="*60)
    print("Starting Tests...")
    print("="*60)
    
    # Test 1: All aspect ratios
    print("\n[Test 1] Testing all aspect ratios...")
    aspect_time = test_all_aspect_ratios()
    
    time.sleep(2)  # Brief pause between tests
    
    # Test 2: Sequential (baseline)
    print("\n[Test 2] Sequential test with mixed aspect ratios...")
    seq_time = test_sequential(num_requests=3)
    
    time.sleep(2)  # Brief pause between tests
    
    # Test 3: Concurrent with 3 workers
    print("\n[Test 3] Concurrent test with mixed aspect ratios...")
    conc_time = test_concurrent(num_requests=3, max_workers=3)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"All Aspect Ratios Test ({len(ASPECT_RATIOS)} requests): {aspect_time:.2f}s")
    print(f"Sequential (3 requests): {seq_time:.2f}s")
    print(f"Concurrent (3 requests, 3 workers): {conc_time:.2f}s")
    print(f"\nNote: With GPU workloads, concurrent requests are queued")
    print(f"      using thread-safe locks to prevent GPU memory conflicts.")
    print(f"      The server can handle multiple requests but processes")
    print(f"      them sequentially on the GPU for stability.")
    print(f"\nAspect Ratios Tested: {', '.join(ASPECT_RATIOS)}")
    print("="*60)
