#!/usr/bin/env python3
"""
Simple ComfyUI Batch Example

A minimal example showing how to queue 10 images with different seeds.
"""

import json
import urllib.request
import random

SERVER = "127.0.0.1:8188"

# Your workflow JSON (export from ComfyUI: File -> Export (API))
workflow = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 12345,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "masterpiece best quality girl"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "bad hands"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0]
        }
    }
}
"""

# Parse the workflow
prompt = json.loads(workflow)

# Queue 10 images with different seeds
print("Queueing 10 images with different seeds...")
for i in range(10):
    # Change the seed (node 3 is KSampler in this example)
    seed = random.randint(0, 2**31 - 1)
    prompt["3"]["inputs"]["seed"] = seed
    
    # Queue it
    data = json.dumps({"prompt": prompt}).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    
    try:
        urllib.request.urlopen(req)
        print(f"  ✓ Queued image {i+1}/10 (seed: {seed})")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\nDone! Check ComfyUI to see the images generating.")



