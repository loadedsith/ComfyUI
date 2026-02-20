#!/usr/bin/env python3
"""
Systematic KSampler Parameter Testing

Tests KSampler parameters one at a time with a fixed seed.
Each variation includes the parameter name in the output filename.

HOW TO USE:
1. Make sure ComfyUI is running (python3 main.py)
2. Run: python3 script_examples/systematic_test.py [--dry-run] [--size small|medium|large] [--output-dir-mode datetime|execution]

Options:
    --dry-run           Generate HTML gallery from existing images without queueing new prompts
    --size              Size mode: small (1 in 5), medium (1 in 3), large (all) [default: large]
    --output-dir-mode   Output directory mode: datetime or execution [default: datetime]
"""

import json
import urllib.request
import uuid
import os
import glob
import re
import argparse
from datetime import datetime
from pathlib import Path

SERVER_ADDRESS = "127.0.0.1:8188"

# Get script directory to find template and output folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "gallery_template.html")
EXECUTION_COUNTER_FILE = os.path.join(PROJECT_ROOT, ".systematic_test_counter")

# Fixed seed for all tests
FIXED_SEED = 560404072833250

# Your workflow
workflow_json = """
{
  "3": {
    "inputs": {
      "seed": 560404072833250,
      "steps": 100,
      "cfg": 11,
      "sampler_name": "heun",
      "scheduler": "karras",
      "denoise": 0.5,
      "model": ["17", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["25", 0]
    },
    "class_type": "KSampler"
  },
  "6": {
    "inputs": {
      "text": "Mortal Kombat character\\n",
      "clip": ["17", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "different person, different face, different character, new person",
      "clip": ["17", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": ["3", 0],
      "vae": ["17", 2]
    },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "SD1.5",
      "images": ["8", 0]
    },
    "class_type": "SaveImage"
  },
  "17": {
    "inputs": {
      "model_path": "lustify-sdxl-nsfwsfw-v2-sdxl"
    },
    "class_type": "DiffusersLoader"
  },
  "19": {
    "inputs": {
      "image": "Yazmeen_Stylmist-Token.png"
    },
    "class_type": "LoadImage"
  },
  "22": {
    "inputs": {
      "target_width": 512,
      "target_height": 512,
      "padding_color": "white",
      "interpolation": "area",
      "image": ["19", 0]
    },
    "class_type": "ResizeAndPadImage"
  },
  "23": {
    "inputs": {
      "mask": ["19", 1]
    },
    "class_type": "InvertMask"
  },
  "25": {
    "inputs": {
      "pixels": ["22", 0],
      "vae": ["17", 2]
    },
    "class_type": "VAEEncode"
  }
}
"""

# Available samplers (from comfy/samplers.py)
SAMPLERS = [
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddpm", "lcm",
    "ipndm", "deis", "res_multistep", "res_multistep_ancestral",
    "gradient_estimation", "er_sde", "seeds_2", "seeds_3", "sa_solver"
]

# Available schedulers
SCHEDULERS = [
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
    "beta", "normal", "linear_quadratic", "kl_optimal"
]

# CFG values to test (expanded range) - full list
CFG_VALUES_FULL = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Denoise values to test (0.05 to 1.0 in 0.05 increments) - full list
DENOISE_VALUES_FULL = [round(x * 0.05, 2) for x in range(1, 21)]  # 0.05, 0.10, ..., 0.95, 1.0

# Steps values to test (expanded range) - full list
STEPS_VALUES_FULL = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]


def filter_by_size_mode(values, size_mode):
    """
    Filter values based on size mode.
    
    Args:
        values: List of values to filter
        size_mode: "small" (1 in 5), "medium" (1 in 3), "large" (all)
    
    Returns:
        Filtered list of values
    """
    if size_mode == "small":
        # Every 5th value (index 0, 4, 9, 14, ...)
        return [values[i] for i in range(0, len(values), 5)]
    elif size_mode == "medium":
        # Every 3rd value (index 0, 2, 5, 8, ...)
        return [values[i] for i in range(0, len(values), 3)]
    elif size_mode == "large":
        # All values
        return values
    else:
        raise ValueError(f"Unknown size mode: {size_mode}")


def get_output_directory(output_dir_mode):
    """
    Get or create output directory based on mode.
    
    Args:
        output_dir_mode: "datetime" or "execution"
    
    Returns:
        Path to output directory
    """
    if output_dir_mode == "datetime":
        # Use date-time format: YYYYMMDD-HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(BASE_OUTPUT_FOLDER, f"systematic_test_{timestamp}")
    elif output_dir_mode == "execution":
        # Use execution number
        execution_num = get_next_execution_number()
        output_dir = os.path.join(BASE_OUTPUT_FOLDER, f"systematic_test_exec_{execution_num:04d}")
    else:
        raise ValueError(f"Unknown output directory mode: {output_dir_mode}")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_next_execution_number():
    """Get next execution number, incrementing counter file."""
    if os.path.exists(EXECUTION_COUNTER_FILE):
        try:
            with open(EXECUTION_COUNTER_FILE, 'r') as f:
                counter = int(f.read().strip())
        except (ValueError, IOError):
            counter = 0
    else:
        counter = 0
    
    counter += 1
    
    try:
        with open(EXECUTION_COUNTER_FILE, 'w') as f:
            f.write(str(counter))
    except IOError:
        pass  # If we can't write, just return the number
    
    return counter

# Section descriptions (optional - shown as blockquote in HTML)
SECTION_DESCRIPTIONS = {
    "Steps": "The number of denoising steps. More steps generally produce higher quality but take longer. Too many steps can lead to over-processing.",
    "Samplers": "Different algorithms for removing noise. Each has unique characteristics in speed, quality, and style.",
    "Schedulers": "Controls how noise is gradually removed during sampling. Different schedulers affect the noise schedule curve.",
    "CFG": "Classifier-Free Guidance scale. Higher values make the model follow the prompt more closely, but too high can reduce quality and creativity.",
    "Denoise": "The amount of denoising applied. Lower values (0.3-0.5) preserve more of the original image structure, higher values (0.7-1.0) create more dramatic changes. Essential for image-to-image workflows."
}

# Sampler descriptions (optional - shown in card caption)
SAMPLER_DESCRIPTIONS = {
    "euler": "Simple, fast, and reliable. Good default choice.",
    "euler_ancestral": "Euler with ancestral sampling. Adds more variation between runs.",
    "heun": "More accurate than Euler but slower. Good quality/speed balance.",
    "dpm_2": "DPM-Solver-2. Fast and efficient.",
    "dpm_2_ancestral": "DPM-2 with ancestral sampling for more variation.",
    "lms": "Linear Multi-Step. Stable but slower.",
    "dpm_fast": "Very fast but lower quality. Good for quick previews.",
    "dpm_adaptive": "Adaptive step sizing. Automatically adjusts steps.",
    "dpmpp_2s_ancestral": "DPM++ 2S Ancestral. Fast with good quality.",
    "dpmpp_sde": "DPM++ SDE. Stochastic Differential Equation variant.",
    "dpmpp_2m": "DPM++ 2M. Popular choice - fast with excellent quality.",
    "dpmpp_2m_sde": "DPM++ 2M SDE. High quality, slower than 2M.",
    "dpmpp_3m_sde": "DPM++ 3M SDE. Highest quality, slowest.",
    "ddpm": "Denoising Diffusion Probabilistic Model. Original DDPM algorithm.",
    "lcm": "Latent Consistency Model. Designed for very few steps (4-8).",
    "ipndm": "iPNDM. Improved Pseudo Numerical methods.",
    "deis": "DEIS. Diffusion Exponential Integrator Scheme.",
    "res_multistep": "Residual Multi-Step. Good for high-resolution.",
    "res_multistep_ancestral": "Residual Multi-Step with ancestral sampling.",
    "gradient_estimation": "Uses gradient estimation techniques.",
    "er_sde": "Euler-Maruyama SDE. Stochastic variant.",
    "seeds_2": "Seeds-2 algorithm.",
    "seeds_3": "Seeds-3 algorithm.",
    "sa_solver": "SA-Solver. Self-Attention based solver."
}


def queue_prompt(prompt, prompt_id=None):
    """Queue a prompt to ComfyUI."""
    if prompt_id is None:
        prompt_id = str(uuid.uuid4())
    
    p = {
        "prompt": prompt,
        "client_id": str(uuid.uuid4()),
        "prompt_id": prompt_id
    }
    
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        print(f"✓ Queued: {prompt_id[:20]}... (queue #{result.get('number', '?')})")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def create_variation(base_prompt, param_name, param_value, filename_suffix, section_name, output_subdir=None, description=None):
    """
    Create a variation with one parameter changed.
    
    Args:
        base_prompt: Base workflow (will be copied)
        param_name: Parameter name (e.g., "steps", "sampler_name")
        param_value: Value to set
        filename_suffix: Suffix to add to filename (e.g., "steps_20")
        section_name: Section name for grouping (e.g., "steps", "samplers")
        output_subdir: Subdirectory name for output (e.g., "systematic_test_exec_0001")
        description: Optional description for this specific variation (e.g., sampler description)
    """
    prompt = json.loads(json.dumps(base_prompt))  # Deep copy
    
    # Set fixed seed
    prompt["3"]["inputs"]["seed"] = FIXED_SEED
    
    # Change the parameter
    if param_name == "steps":
        prompt["3"]["inputs"]["steps"] = param_value
    elif param_name == "sampler_name":
        prompt["3"]["inputs"]["sampler_name"] = param_value
    elif param_name == "scheduler":
        prompt["3"]["inputs"]["scheduler"] = param_value
    elif param_name == "cfg":
        prompt["3"]["inputs"]["cfg"] = param_value
    elif param_name == "denoise":
        prompt["3"]["inputs"]["denoise"] = param_value
    
    # Update filename to include parameter and subdirectory
    # filename_prefix can include subfolder path (e.g., "subfolder/SD1.5_steps_20")
    if output_subdir:
        prompt["9"]["inputs"]["filename_prefix"] = f"{output_subdir}/SD1.5_{filename_suffix}"
    else:
        prompt["9"]["inputs"]["filename_prefix"] = f"SD1.5_{filename_suffix}"
    
    return {
        "prompt": prompt,
        "param_name": param_name,
        "param_value": param_value,
        "filename_suffix": filename_suffix,
        "section_name": section_name,
        "description": description  # Optional description for this card
    }


def find_images_in_output(output_dir):
    """Scan output folder for images matching our naming pattern."""
    images = {}
    pattern = os.path.join(output_dir, "SD1.5_*.png")
    
    for img_path in glob.glob(pattern):
        filename = os.path.basename(img_path)
        # Extract parameter suffix from filename like "SD1.5_steps_20_00001_.png"
        match = re.match(r"SD1\.5_(.+?)_\d+_\.png", filename)
        if match:
            suffix = match.group(1)
            images[suffix] = img_path
    
    return images


def generate_html_gallery(variations, images_dict, output_dir, dry_run=False):
    """Generate HTML gallery from variations and found images."""
    # Load template
    try:
        with open(TEMPLATE_FILE, 'r') as f:
            template = f.read()
    except FileNotFoundError:
        print(f"✗ Template file not found: {TEMPLATE_FILE}")
        return None
    
    # Group variations by section
    sections = {}
    for var in variations:
        section = var["section_name"]
        if section not in sections:
            sections[section] = []
        sections[section].append(var)
    
    # Generate gallery content
    gallery_html = ""
    
    for section_name, section_vars in sections.items():
        # Section header
        gallery_html += f'<div class="gallery-section">\n'
        gallery_html += f'<div class="section-title">{section_name.upper()}</div>\n'
        
        # Add section description if available
        section_desc = SECTION_DESCRIPTIONS.get(section_name)
        if section_desc:
            gallery_html += f'<blockquote class="section-description">{section_desc}</blockquote>\n'
        
        # Create cards container
        gallery_html += '<div class="cards-container">\n'
        
        # Add cards
        for var in section_vars:
            suffix = var["filename_suffix"]
            param_value = var["param_value"]
            description = var.get("description")  # Optional card description
            
            # Find matching image
            img_path = images_dict.get(suffix)
            
            if img_path:
                # HTML file is in OUTPUT_FOLDER, so just use filename
                img_filename = os.path.basename(img_path)
                gallery_html += '<div class="image-card">\n'
                gallery_html += '<div class="image-wrapper">\n'
                gallery_html += f'<img src="{img_filename}" alt="{suffix}">\n'
                gallery_html += '</div>\n'
                gallery_html += f'<div class="card-caption">{param_value}</div>\n'
                # Add description if available
                if description:
                    gallery_html += f'<div class="card-description">{description}</div>\n'
                gallery_html += f'<div class="card-label">{suffix}</div>\n'
                gallery_html += '</div>\n'
            else:
                gallery_html += '<div class="missing-card">\n'
                gallery_html += '<div>Image not found</div>\n'
                gallery_html += f'<div class="card-caption">{param_value}</div>\n'
                # Add description if available
                if description:
                    gallery_html += f'<div class="card-description">{description}</div>\n'
                gallery_html += f'<div class="card-label">{suffix}</div>\n'
                gallery_html += '</div>\n'
        
        gallery_html += '</div>\n'  # Close cards-container
        gallery_html += '</div>\n\n'  # Close gallery-section
    
    # Replace template placeholders
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_images = len([v for v in variations if v["filename_suffix"] in images_dict])
    missing_images = len(variations) - total_images
    
    template = template.replace("{{GENERATION_DATE}}", now)
    template = template.replace("{{FIXED_SEED}}", str(FIXED_SEED))
    template = template.replace("{{TOTAL_VARIATIONS}}", str(len(variations)))
    template = template.replace("{{TOTAL_IMAGES}}", str(total_images))
    template = template.replace("{{IMAGES_FOUND}}", str(total_images))
    template = template.replace("{{MISSING_IMAGES}}", str(missing_images))
    template = template.replace("{{GALLERY_CONTENT}}", gallery_html)
    
    # Save HTML file in the output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    html_filename = f"gallery-{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(template)
    
    return html_path


def create_all_variations(base_prompt, size_mode="large", output_subdir=None):
    """
    Create all variation definitions.
    
    Args:
        base_prompt: Base workflow JSON
        size_mode: "small" (1 in 5), "medium" (1 in 3), "large" (all)
        output_subdir: Subdirectory name for output (e.g., "systematic_test_exec_0001")
    """
    variations = []
    
    # Filter values based on size mode
    steps_values = filter_by_size_mode(STEPS_VALUES_FULL, size_mode)
    cfg_values = filter_by_size_mode(CFG_VALUES_FULL, size_mode)
    denoise_values = filter_by_size_mode(DENOISE_VALUES_FULL, size_mode)
    
    # Samplers and schedulers are always all (they're not numeric ranges)
    
    # ========================================================================
    # TEST 1: Steps (filtered by size mode)
    # ========================================================================
    for steps in steps_values:
        filename_suffix = f"steps_{steps}"
        var = create_variation(base_prompt, "steps", steps, filename_suffix, "Steps", output_subdir=output_subdir)
        variations.append(var)
    
    # ========================================================================
    # TEST 2: Samplers (with steps fixed at 20)
    # ========================================================================
    for sampler in SAMPLERS:
        clean_name = sampler.replace("_", "-")
        filename_suffix = f"sampler_{clean_name}"
        # Get optional description for this sampler
        sampler_desc = SAMPLER_DESCRIPTIONS.get(sampler)
        var = create_variation(base_prompt, "sampler_name", sampler, filename_suffix, "Samplers", output_subdir=output_subdir, description=sampler_desc)
        var["prompt"]["3"]["inputs"]["steps"] = 20
        variations.append(var)
    
    # ========================================================================
    # TEST 3: Schedulers (with steps=20, sampler=heun)
    # ========================================================================
    for scheduler in SCHEDULERS:
        clean_name = scheduler.replace("_", "-")
        filename_suffix = f"scheduler_{clean_name}"
        var = create_variation(base_prompt, "scheduler", scheduler, filename_suffix, "Schedulers", output_subdir=output_subdir)
        var["prompt"]["3"]["inputs"]["steps"] = 20
        var["prompt"]["3"]["inputs"]["sampler_name"] = "heun"
        variations.append(var)
    
    # ========================================================================
    # TEST 4: CFG (with steps=20, sampler=heun, scheduler=karras)
    # ========================================================================
    for cfg in cfg_values:
        filename_suffix = f"cfg_{cfg}"
        var = create_variation(base_prompt, "cfg", cfg, filename_suffix, "CFG", output_subdir=output_subdir)
        var["prompt"]["3"]["inputs"]["steps"] = 20
        var["prompt"]["3"]["inputs"]["sampler_name"] = "heun"
        var["prompt"]["3"]["inputs"]["scheduler"] = "karras"
        variations.append(var)
    
    # ========================================================================
    # TEST 5: Denoise (with steps=20, sampler=heun, scheduler=karras, cfg=11)
    # ========================================================================
    for denoise in denoise_values:
        # Format denoise value for filename (0.05 -> "0-05", 0.1 -> "0-1", 1.0 -> "1-0")
        denoise_str = f"{denoise:.2f}".replace(".", "-")
        filename_suffix = f"denoise_{denoise_str}"
        var = create_variation(base_prompt, "denoise", denoise, filename_suffix, "Denoise", output_subdir=output_subdir)
        var["prompt"]["3"]["inputs"]["steps"] = 20
        var["prompt"]["3"]["inputs"]["sampler_name"] = "heun"
        var["prompt"]["3"]["inputs"]["scheduler"] = "karras"
        var["prompt"]["3"]["inputs"]["cfg"] = 11
        variations.append(var)
    
    return variations


def main():
    parser = argparse.ArgumentParser(description="Systematic KSampler Parameter Testing")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Generate HTML gallery from existing images without queueing new prompts")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="large",
                       help="Size mode: small (1 in 5), medium (1 in 3), large (all) [default: large]")
    parser.add_argument("--output-dir-mode", choices=["datetime", "execution"], default="datetime",
                       help="Output directory mode: datetime or execution [default: datetime]")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Specific output directory to use (overrides --output-dir-mode)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Systematic KSampler Parameter Testing")
    print("=" * 70)
    print(f"Fixed seed: {FIXED_SEED}")
    print(f"Size mode: {args.size}")
    print(f"Output directory mode: {args.output_dir_mode}")
    if args.dry_run:
        print("Mode: DRY RUN (HTML generation only)")
    print()
    
    # Get or create output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = get_output_directory(args.output_dir_mode)
    
    # Get subdirectory name (relative to base output folder) for use in filename_prefix
    output_subdir = os.path.basename(output_dir)
    
    print(f"Output directory: {output_dir}")
    print(f"Output subdirectory: {output_subdir}")
    print()
    
    # Parse workflow
    try:
        base_prompt = json.loads(workflow_json)
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing workflow: {e}")
        return
    
    # Create all variations (filtered by size mode)
    print("Creating variation definitions...")
    variations = create_all_variations(base_prompt, size_mode=args.size, output_subdir=output_subdir)
    print(f"✓ Created {len(variations)} variation definitions")
    
    # Show breakdown by parameter
    steps_count = len(filter_by_size_mode(STEPS_VALUES_FULL, args.size))
    cfg_count = len(filter_by_size_mode(CFG_VALUES_FULL, args.size))
    denoise_count = len(filter_by_size_mode(DENOISE_VALUES_FULL, args.size))
    print(f"  Steps: {steps_count} variations")
    print(f"  Samplers: {len(SAMPLERS)} variations")
    print(f"  Schedulers: {len(SCHEDULERS)} variations")
    print(f"  CFG: {cfg_count} variations")
    print(f"  Denoise: {denoise_count} variations")
    print()
    
    if args.dry_run:
        # Dry run: just generate HTML from existing images
        print("Scanning output folder for images...")
        images_dict = find_images_in_output(output_dir)
        print(f"✓ Found {len(images_dict)} matching images")
        print()
        
        print("Generating HTML gallery...")
        html_path = generate_html_gallery(variations, images_dict, output_dir, dry_run=True)
        
        if html_path:
            print(f"✓ HTML gallery generated: {html_path}")
            print(f"  Open it in your browser to view the comparison gallery!")
        else:
            print("✗ Failed to generate HTML gallery")
    else:
        # Normal mode: queue prompts and generate HTML
        print("=" * 70)
        print(f"Total variations: {len(variations)}")
        print("=" * 70)
        print()
        
        queued = []
        for i, variation in enumerate(variations, 1):
            name = variation["filename_suffix"]
            prompt = variation["prompt"]
            prompt_id = f"test_{name}_{uuid.uuid4().hex[:8]}"
            
            print(f"[{i:3d}/{len(variations)}] {name}")
            result = queue_prompt(prompt, prompt_id)
            
            if result:
                queued.append(name)
        
        # Summary
        print()
        print("=" * 70)
        print(f"✓ Successfully queued {len(queued)}/{len(variations)} variations")
        print("=" * 70)
        print()
        print("Summary by parameter:")
        print(f"  Steps: {steps_count} variations")
        print(f"  Samplers: {len(SAMPLERS)} variations")
        print(f"  Schedulers: {len(SCHEDULERS)} variations")
        print(f"  CFG: {cfg_count} variations")
        print(f"  Denoise: {denoise_count} variations")
        print()
        print(f"Output directory: {output_dir}")
        print()
        
        # Always try to generate HTML from any existing images
        print("Scanning for existing images...")
        images_dict = find_images_in_output(output_dir)
        if images_dict:
            print(f"Found {len(images_dict)} existing images, generating HTML...")
            html_path = generate_html_gallery(variations, images_dict, output_dir, dry_run=False)
            if html_path:
                print(f"✓ HTML gallery generated: {html_path}")
                print(f"  Open it in your browser to view the comparison gallery!")
        else:
            print("No matching images found yet.")
            print()
            print("After images are generated, run with --dry-run to create HTML gallery:")
            print(f"  python3 script_examples/systematic_test.py --dry-run --output-dir \"{output_dir}\"")


if __name__ == "__main__":
    main()

