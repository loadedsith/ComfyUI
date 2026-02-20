#!/usr/bin/env python3
"""
ComfyUI API - Retrieve Prompts and Images

Demonstrates how to retrieve:
1. The prompt/workflow JSON that was used
2. Output images
3. Input images (if available in workflow)

HOW TO USE:
    python3 script_examples/retrieve_prompt_and_images.py [prompt_id]
    
If no prompt_id is provided, it will list recent history.
"""

import json
import urllib.request
import urllib.parse
import sys
import os
from pathlib import Path

SERVER_ADDRESS = "127.0.0.1:8188"


def get_history(prompt_id=None, max_items=None):
    """
    Get history from ComfyUI.
    
    Args:
        prompt_id: Specific prompt ID, or None for all history
        max_items: Maximum number of items to return (for all history)
    
    Returns:
        Dictionary with prompt_id as keys, containing:
        - "prompt": The full workflow JSON that was used
        - "outputs": Dictionary of node outputs (including images)
        - "status": Execution status
    """
    if prompt_id:
        url = f"http://{SERVER_ADDRESS}/history/{prompt_id}"
    else:
        url = f"http://{SERVER_ADDRESS}/history"
        if max_items:
            url += f"?max_items={max_items}"
    
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"✗ Error retrieving history: {e}")
        return {}


def get_image(filename, subfolder="", folder_type="output"):
    """
    Retrieve an image from ComfyUI.
    
    Args:
        filename: Image filename
        subfolder: Subfolder (usually empty for outputs)
        folder_type: Type of folder ("input", "output", "temp")
    
    Returns:
        Image data as bytes, or None if error
    """
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    url = f"http://{SERVER_ADDRESS}/view?{url_values}"
    
    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except Exception as e:
        print(f"✗ Error retrieving image {filename}: {e}")
        return None


def extract_prompt_info(prompt_json):
    """
    Extract useful information from a prompt/workflow JSON.
    
    Returns a dictionary with extracted parameters.
    """
    info = {
        "model": None,
        "positive_prompt": None,
        "negative_prompt": None,
        "seed": None,
        "steps": None,
        "cfg": None,
        "sampler": None,
        "scheduler": None,
        "denoise": None,
        "width": None,
        "height": None,
        "input_image": None,
    }
    
    # Find KSampler node (usually node "3")
    for node_id, node_data in prompt_json.items():
        if node_data.get("class_type") == "KSampler":
            inputs = node_data.get("inputs", {})
            info["seed"] = inputs.get("seed")
            info["steps"] = inputs.get("steps")
            info["cfg"] = inputs.get("cfg")
            info["sampler"] = inputs.get("sampler_name")
            info["scheduler"] = inputs.get("scheduler")
            info["denoise"] = inputs.get("denoise")
            break
    
    # Find CLIPTextEncode nodes (prompts)
    for node_id, node_data in prompt_json.items():
        if node_data.get("class_type") == "CLIPTextEncode":
            text = node_data.get("inputs", {}).get("text", "")
            # Usually positive prompt comes before negative
            if info["positive_prompt"] is None:
                info["positive_prompt"] = text
            else:
                info["negative_prompt"] = text
    
    # Find model loader
    for node_id, node_data in prompt_json.items():
        if node_data.get("class_type") == "CheckpointLoaderSimple":
            info["model"] = node_data.get("inputs", {}).get("ckpt_name")
        elif node_data.get("class_type") == "DiffusersLoader":
            info["model"] = node_data.get("inputs", {}).get("model_path")
    
    # Find image dimensions
    for node_id, node_data in prompt_json.items():
        if node_data.get("class_type") == "EmptyLatentImage":
            inputs = node_data.get("inputs", {})
            info["width"] = inputs.get("width")
            info["height"] = inputs.get("height")
        elif node_id == "19":  # Common node ID for LoadImage in img2img workflows
            info["input_image"] = node_data.get("inputs", {}).get("image")
    
    return info


def save_image(image_data, filename, output_dir="retrieved_images"):
    """Save image data to a file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)
    return filepath


def main():
    if len(sys.argv) > 1:
        prompt_id = sys.argv[1]
        print(f"Retrieving prompt ID: {prompt_id}")
        history = get_history(prompt_id)
    else:
        print("No prompt ID provided. Retrieving recent history...")
        history = get_history(max_items=10)
        if history:
            print("\nRecent prompt IDs:")
            for pid in list(history.keys())[-10:]:
                print(f"  - {pid}")
            print("\nRun with a prompt ID to see details:")
            print(f"  python3 {sys.argv[0]} <prompt_id>")
            return
        else:
            print("No history found.")
            return
    
    if not history:
        print("No history found for the given prompt ID.")
        return
    
    # Process each prompt in history
    for prompt_id, prompt_data in history.items():
        print(f"\n{'='*70}")
        print(f"Prompt ID: {prompt_id}")
        print(f"{'='*70}")
        
        # Get the prompt/workflow JSON
        # The prompt is stored as a tuple: (number, prompt_id, prompt_json, extra_data, outputs, sensitive)
        prompt_tuple = prompt_data.get("prompt", [])
        if isinstance(prompt_tuple, (list, tuple)) and len(prompt_tuple) > 2:
            prompt_json = prompt_tuple[2]  # The actual workflow is at index 2
        elif isinstance(prompt_tuple, dict):
            # Sometimes it might already be the JSON
            prompt_json = prompt_tuple
        else:
            print("No prompt JSON found.")
            continue
        
        # Extract useful info
        info = extract_prompt_info(prompt_json)
        
        print("\nExtracted Parameters:")
        print(f"  Model: {info['model']}")
        print(f"  Positive Prompt: {info['positive_prompt']}")
        print(f"  Negative Prompt: {info['negative_prompt']}")
        print(f"  Seed: {info['seed']}")
        print(f"  Steps: {info['steps']}")
        print(f"  CFG: {info['cfg']}")
        print(f"  Sampler: {info['sampler']}")
        print(f"  Scheduler: {info['scheduler']}")
        print(f"  Denoise: {info['denoise']}")
        print(f"  Dimensions: {info['width']}x{info['height']}")
        if info['input_image']:
            print(f"  Input Image: {info['input_image']}")
        
        # Get outputs (images)
        outputs = prompt_data.get("outputs", {})
        print(f"\nOutputs found: {len(outputs)} node(s)")
        
        saved_count = 0
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                print(f"\n  Node {node_id} images:")
                for img_info in node_output["images"]:
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    folder_type = img_info.get("type", "output")
                    
                    print(f"    - {filename} (type: {folder_type})")
                    
                    # Retrieve and save the image
                    image_data = get_image(filename, subfolder, folder_type)
                    if image_data:
                        saved_path = save_image(image_data, filename)
                        print(f"      ✓ Saved to: {saved_path}")
                        saved_count += 1
        
        # Save the full prompt JSON
        prompt_file = f"retrieved_images/prompt_{prompt_id}.json"
        os.makedirs("retrieved_images", exist_ok=True)
        with open(prompt_file, 'w') as f:
            json.dump(prompt_json, f, indent=2)
        print(f"\n✓ Full prompt JSON saved to: {prompt_file}")
        
        # Status
        status = prompt_data.get("status", {})
        if status:
            print(f"\nStatus: {status.get('status_str', 'unknown')}")
            if status.get('messages'):
                print("Messages:")
                for msg in status['messages']:
                    print(f"  - {msg}")
        
        print(f"\n✓ Retrieved {saved_count} image(s)")


if __name__ == "__main__":
    main()

