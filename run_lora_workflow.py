#!/usr/bin/env python3
"""
Run IMG Refiner workflow for each artistic SDXL LoRA and generate HTML gallery
"""

import json
import urllib.request
import urllib.parse
import time
import os
import sys
import random
from pathlib import Path

SERVER_ADDRESS = "127.0.0.1:8188"
WORKFLOW_FILE = "user/default/workflows/IMG Refiner.json"
LORAS_DIR = "models/loras"
OUTPUT_DIR = "lora_gallery_output"
HTML_FILE = "lora_gallery.html"

# Artistic LoRAs to test (excluding non-artistic ones)
ARTISTIC_LORAS = [
    "vangogh_art_style_sdxl.safetensors",
    "indie_comic_art_style_sdxl.safetensors",
    "indie_art_style_sdxl.safetensors",
    "dynamic_anatomy_sdxl.safetensors",
]


def load_workflow():
    """Load the workflow JSON file and convert to API format."""
    workflow_path = Path(WORKFLOW_FILE)
    if not workflow_path.exists():
        print(f"✗ Workflow file not found: {WORKFLOW_FILE}")
        sys.exit(1)

    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)

    # Convert nodes array to dictionary format (API format)
    if "nodes" in workflow_data:
        nodes_dict = {}
        links = workflow_data.get("links", [])

        # Build a map of target_node -> input_name -> [source_node, output_index]
        link_map = {}
        for link in links:
            # Link format: [link_id, source_node_id, source_slot, target_node_id, target_slot, type]
            if len(link) >= 5:
                source_id = str(link[1])  # source node is at index 1
                target_id = str(link[3])   # target node is at index 3
                target_slot = link[4]      # target slot is at index 4

                if target_id not in link_map:
                    link_map[target_id] = {}
                # Find the input name for this slot
                target_node = next((n for n in workflow_data["nodes"] if str(n["id"]) == target_id), None)
                if target_node:
                    inputs = target_node.get("inputs", [])
                    if target_slot < len(inputs):
                        input_name = inputs[target_slot].get("name")
                        if input_name:
                            source_slot = link[2]  # source output slot
                            link_map[target_id][input_name] = [source_id, source_slot]

        for node in workflow_data["nodes"]:
            node_id = str(node["id"])
            node_dict = {
                "class_type": node["type"],
                "inputs": {}
            }

            # Process inputs - convert array format to dictionary
            widgets_values = node.get("widgets_values", [])
            widget_idx = 0

            # Count total widget inputs
            total_widget_inputs = sum(1 for inp in node.get("inputs", [])
                                     if "widget" in inp and inp.get("widget"))

            for inp in node.get("inputs", []):
                inp_name = inp.get("name")
                if inp_name:
                    # Check if this input is linked to another node
                    if node_id in link_map and inp_name in link_map[node_id]:
                        node_dict["inputs"][inp_name] = link_map[node_id][inp_name]
                    # Check if this input has a widget (user-editable value)
                    elif "widget" in inp and inp["widget"]:
                        widget_name = inp["widget"].get("name")
                        if widget_name and widget_idx < len(widgets_values):
                            value = widgets_values[widget_idx]

                            # Handle special case: if widgets_values has more values than widget inputs,
                            # and we're at the last widget input (denoise), use the last value instead
                            if inp_name == "denoise" and len(widgets_values) > total_widget_inputs:
                                if widget_idx == total_widget_inputs - 1:
                                    # Use the last value in widgets_values as the actual denoise
                                    value = widgets_values[-1]

                            # Handle special UI values
                            if inp_name == "steps" and value == "randomize":
                                value = 20  # Default steps value
                            elif inp_name == "seed" and value == "randomize":
                                import random
                                value = random.randint(0, 2**31 - 1)
                            elif inp_name == "denoise":
                                # Ensure denoise is a float
                                if isinstance(value, str):
                                    try:
                                        value = float(value)
                                    except:
                                        value = 1.0
                                elif not isinstance(value, (int, float)):
                                    value = 1.0
                            elif inp_name == "sampler_name" and isinstance(value, int):
                                # Convert integer to string sampler name
                                samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                                           "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                                           "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                                           "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm"]
                                if 0 <= value < len(samplers):
                                    value = samplers[value]
                                else:
                                    value = "euler"  # Default
                            elif inp_name == "scheduler":
                                # SDXL refiner only supports specific schedulers
                                valid_schedulers = ['simple', 'sgm_uniform', 'karras', 'exponential',
                                                   'ddim_uniform', 'beta', 'normal', 'linear_quadratic', 'kl_optimal']
                                if value not in valid_schedulers:
                                    # Map common invalid values to valid ones
                                    scheduler_map = {
                                        'dpmpp_2m': 'karras',
                                        'euler': 'simple',
                                        'normal': 'normal',
                                        'karras': 'karras'
                                    }
                                    value = scheduler_map.get(value, 'simple')  # Default to 'simple'
                            # Use the widget name (which matches the input name)
                            node_dict["inputs"][inp_name] = value
                            widget_idx += 1

            # Don't include widgets_values in API format - API only needs inputs
            # Store widgets_values separately only for our internal use
            # nodes_dict[node_id]["widgets_values"] = widgets_values  # Not needed for API
            nodes_dict[node_id] = node_dict

        return nodes_dict
    else:
        # Already in dictionary format
        return workflow_data


def queue_prompt(prompt):
    """Queue a prompt/workflow to ComfyUI."""
    data = json.dumps({"prompt": prompt}).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            return result.get("prompt_id")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_json = json.loads(error_body)
            if "node_errors" in error_json:
                for node_id, errors in error_json["node_errors"].items():
                    print(f"  Node {node_id} errors: {errors}")
            else:
                print(f"✗ HTTP Error {e.code}: {error_body[:500]}")
        except:
            print(f"✗ HTTP Error {e.code}: {error_body[:500]}")
        return None
    except Exception as e:
        print(f"✗ Error queueing prompt: {e}")
        return None


def get_history(prompt_id):
    """Get history for a specific prompt ID."""
    url = f"http://{SERVER_ADDRESS}/history/{prompt_id}"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"✗ Error retrieving history: {e}")
        return {}


def wait_for_completion(prompt_id, timeout=300):
    """Wait for a prompt to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            status_str = status.get("status_str", "")
            if status_str == "success":
                return True
            elif status_str == "error":
                return False
        time.sleep(2)
    return False


def get_image(filename, subfolder="", folder_type="output"):
    """Retrieve an image from ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    url = f"http://{SERVER_ADDRESS}/view?{url_values}"

    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except Exception as e:
        print(f"✗ Error retrieving image {filename}: {e}")
        return None


def get_output_images(prompt_id):
    """Get output images for a completed prompt."""
    history = get_history(prompt_id)
    if prompt_id not in history:
        return []

    outputs = history[prompt_id].get("outputs", {})
    images = []

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for img_info in node_output["images"]:
                images.append({
                    "filename": img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type": img_info.get("type", "output")
                })

    return images


def generate_html_gallery(results):
    """Generate HTML gallery from results."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SDXL LoRA Gallery</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }
        .card-image {
            width: 100%;
            height: 300px;
            object-fit: cover;
            cursor: pointer;
            transition: opacity 0.3s ease;
        }
        .card-image:hover {
            opacity: 0.9;
        }
        .card-label {
            padding: 20px;
            text-align: center;
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }
        .status {
            text-align: center;
            color: white;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        .error {
            background: #ff6b6b;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 SDXL LoRA Gallery</h1>
        <div class="status">Generated images from artistic SDXL LoRAs</div>
        <div class="gallery">
"""

    for lora_name, result in results.items():
        if result.get("error"):
            html_content += f"""
            <div class="card">
                <div class="error">{lora_name}: {result['error']}</div>
            </div>
"""
        elif result.get("image_path"):
            image_filename = os.path.basename(result["image_path"])
            html_content += f"""
            <div class="card">
                <a href="{image_filename}" target="_blank">
                    <img src="{image_filename}" alt="{lora_name}" class="card-image">
                </a>
                <div class="card-label">{lora_name.replace('_', ' ').replace('.safetensors', '')}</div>
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    return html_content


def main():
    print("=" * 70)
    print("SDXL LoRA Workflow Runner")
    print("=" * 70)

    # Check if ComfyUI server is running
    try:
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/system_stats", timeout=5) as response:
            print("✓ ComfyUI server is running")
    except Exception as e:
        print(f"✗ Cannot connect to ComfyUI server at {SERVER_ADDRESS}")
        print("  Make sure ComfyUI is running!")
        sys.exit(1)

    # Load workflow
    print(f"\nLoading workflow: {WORKFLOW_FILE}")
    workflow = load_workflow()
    print("✓ Workflow loaded")

    # Check for common issues in the workflow
    print("\nChecking workflow for issues...")
    workflow_path = Path(WORKFLOW_FILE)
    with open(workflow_path, 'r') as f:
        workflow_raw = json.load(f)

    issues_found = []

    # Check refiner positive prompt (Node 30)
    node30 = next((n for n in workflow_raw["nodes"] if n["id"] == 30), None)
    if node30:
        refiner_pos_text = node30.get("widgets_values", [""])[0] if node30.get("widgets_values") else ""
        if not refiner_pos_text or refiner_pos_text.strip() == "":
            issues_found.append("⚠️  REFINER POSITIVE PROMPT (Node 30) IS EMPTY! The refiner needs a positive prompt.")

    # Check refiner denoise
    node36 = next((n for n in workflow_raw["nodes"] if n["id"] == 36), None)
    if node36:
        refiner_denoise = node36.get("widgets_values", [])[-1] if node36.get("widgets_values") else 1.0
        if isinstance(refiner_denoise, (int, float)) and refiner_denoise < 0.3:
            issues_found.append(f"⚠️  REFINER DENOISE is very low ({refiner_denoise}). Consider increasing to 0.3-0.5 for better results.")

    # Check base negative prompt
    node7 = next((n for n in workflow_raw["nodes"] if n["id"] == 7), None)
    if node7:
        base_neg_text = node7.get("widgets_values", [""])[0] if node7.get("widgets_values") else ""
        if "dicks" in base_neg_text.lower() or "penis" in base_neg_text.lower():
            issues_found.append("⚠️  BASE NEGATIVE PROMPT (Node 7) contains problematic text that may cause issues.")

    if issues_found:
        print("\n⚠️  WORKFLOW ISSUES DETECTED:")
        for issue in issues_found:
            print(f"  {issue}")
        print("\n  These issues may cause:")
        print("    - Ignored prompts")
        print("    - Doubled/weird body parts")
        print("    - Poor image quality")
        print("\n  Please fix these in ComfyUI and save the workflow before running again.")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting. Please fix the workflow issues first.")
            sys.exit(1)
    else:
        print("✓ No obvious issues detected")

    # Filter LoRAs that actually exist
    loras_dir = Path(LORAS_DIR)
    available_loras = []
    for lora in ARTISTIC_LORAS:
        lora_path = loras_dir / lora
        if lora_path.exists():
            available_loras.append(lora)
        else:
            print(f"⚠ LoRA not found: {lora}")

    if not available_loras:
        print("✗ No LoRAs found!")
        sys.exit(1)

    print(f"\nFound {len(available_loras)} LoRA(s) to process:")
    for lora in available_loras:
        print(f"  - {lora}")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # Process each LoRA
    results = {}
    for i, lora_name in enumerate(available_loras, 1):
        print(f"\n[{i}/{len(available_loras)}] Processing: {lora_name}")

        # Create a copy of the workflow
        workflow_copy = json.loads(json.dumps(workflow))

        # Fix empty refiner positive prompt if needed (Node 30)
        node30 = workflow_copy.get("30", {})
        if node30.get("class_type") == "CLIPTextEncode":
            refiner_text = node30.get("inputs", {}).get("text", "")
            if not refiner_text or refiner_text.strip() == "":
                # Use a generic refiner prompt
                node30["inputs"]["text"] = "high quality, detailed, refined"
                if "widgets_values" in node30:
                    node30["widgets_values"][0] = "high quality, detailed, refined"

        # Update LoraLoader node (id 43)
        lora_node_id = None
        for node_id, node_data in workflow_copy.items():
            if node_data.get("class_type") == "LoraLoader":
                lora_node_id = node_id
                break

        if lora_node_id:
            # Update the LoRA name in inputs (API format)
            node = workflow_copy[lora_node_id]
            if "inputs" in node and "lora_name" in node["inputs"]:
                node["inputs"]["lora_name"] = lora_name
            # Also update widgets_values for reference
            if "widgets_values" in node and len(node["widgets_values"]) > 0:
                node["widgets_values"][0] = lora_name

        # Update SaveImage filename_prefix (node 9)
        save_node_id = None
        for node_id, node_data in workflow_copy.items():
            if node_data.get("class_type") == "SaveImage":
                save_node_id = node_id
                break

        if save_node_id:
            lora_display_name = lora_name.replace("_", " ").replace(".safetensors", "")
            node = workflow_copy[save_node_id]
            if "inputs" in node and "filename_prefix" in node["inputs"]:
                node["inputs"]["filename_prefix"] = f"LoRA_{lora_display_name}"
            # Also update widgets_values for reference
            if "widgets_values" in node and len(node["widgets_values"]) > 0:
                node["widgets_values"][0] = f"LoRA_{lora_display_name}"

        # Randomize seeds for each run to ensure different outputs
        # Find all KSampler nodes and randomize their seeds
        new_seed = random.randint(0, 2**31 - 1)
        for node_id, node_data in workflow_copy.items():
            if node_data.get("class_type") == "KSampler":
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    # Generate a random seed for each run
                    node_data["inputs"]["seed"] = new_seed
                # Also update widgets_values if present (seed is usually first widget)
                if "widgets_values" in node_data and len(node_data["widgets_values"]) > 0:
                    node_data["widgets_values"][0] = new_seed

                # Fix refiner settings to prevent double body parts
                # Refiner is typically node 36, base is node 3
                if node_id == "36":  # Refiner KSampler
                    # Lower refiner settings to prevent over-processing
                    if "inputs" in node_data:
                        # Reduce denoise to prevent double body parts
                        if node_data["inputs"].get("denoise", 1.0) > 0.3:
                            node_data["inputs"]["denoise"] = 0.25  # Lower denoise
                        # Reduce steps for refiner (10-15 is usually enough)
                        if node_data["inputs"].get("steps", 20) > 15:
                            node_data["inputs"]["steps"] = 12
                        # Lower CFG for refiner
                        if node_data["inputs"].get("cfg", 15) > 10:
                            node_data["inputs"]["cfg"] = 7

        # Debug: Show prompt values being used
        node28 = workflow_copy.get("28", {})
        node7 = workflow_copy.get("7", {})
        node30 = workflow_copy.get("30", {})
        node31 = workflow_copy.get("31", {})

        base_pos = node28.get("inputs", {}).get("text", "")[:80] if node28.get("inputs", {}).get("text") else ""
        base_neg = node7.get("inputs", {}).get("text", "")[:80] if node7.get("inputs", {}).get("text") else ""
        refiner_pos = node30.get("inputs", {}).get("text", "")[:80] if node30.get("inputs", {}).get("text") else ""
        refiner_neg = node31.get("inputs", {}).get("text", "")[:80] if node31.get("inputs", {}).get("text") else ""

        print(f"  Base prompt: {base_pos}...")
        print(f"  Refiner prompt: {refiner_pos}...")

        # Save converted workflow for debugging (first LoRA only)
        if i == 1:
            debug_file = output_path / f"debug_workflow_{lora_name.replace('.safetensors', '')}.json"
            with open(debug_file, 'w') as f:
                json.dump(workflow_copy, f, indent=2)
            print(f"  Debug: Saved converted workflow to {debug_file}")

        # Queue the workflow
        prompt_id = queue_prompt(workflow_copy)
        if not prompt_id:
            results[lora_name] = {"error": "Failed to queue prompt"}
            continue

        print(f"  ✓ Queued (prompt_id: {prompt_id})")

        # Wait for completion
        print("  Waiting for completion...", end="", flush=True)
        if wait_for_completion(prompt_id):
            print(" ✓")

            # Get output images
            images = get_output_images(prompt_id)
            if images:
                # Download the first image
                img_info = images[0]
                image_data = get_image(img_info["filename"], img_info["subfolder"], img_info["type"])

                if image_data:
                    # Save image
                    image_filename = f"{lora_name.replace('.safetensors', '')}_{img_info['filename']}"
                    image_path = output_path / image_filename
                    with open(image_path, 'wb') as f:
                        f.write(image_data)

                    results[lora_name] = {
                        "image_path": str(image_path),
                        "prompt_id": prompt_id
                    }
                    print(f"  ✓ Image saved: {image_filename}")
                else:
                    results[lora_name] = {"error": "Failed to retrieve image"}
            else:
                results[lora_name] = {"error": "No images in output"}
        else:
            print(" ✗")
            results[lora_name] = {"error": "Timeout waiting for completion"}

    # Generate HTML gallery
    print(f"\nGenerating HTML gallery...")
    html_content = generate_html_gallery(results)
    html_path = output_path / HTML_FILE

    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"✓ HTML gallery saved: {html_path}")
    print(f"\n{'=' * 70}")
    print("Done! Open the HTML file in your browser to view the gallery.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

