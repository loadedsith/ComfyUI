#!/usr/bin/env python3
"""
ComfyUI Batch Variations Script

This script demonstrates how to queue multiple image generations with different settings.
You can vary: seed, CFG, steps, sampler, scheduler, prompts, etc.

HOW TO USE:
Option 1 - From History (Recommended):
    python3 batch_variations.py --from-history [prompt_id]
    # If no prompt_id, lists recent workflows to choose from

Option 2 - Manual Workflow:
    python3 batch_variations.py --workflow-file workflow.json
    # Or edit DEFAULT_WORKFLOW variable below

Option 3 - Inline JSON:
    python3 batch_variations.py --workflow-json '{"3": {...}}'
"""

import json
import urllib.request
import urllib.parse
import random
import sys
import argparse
import uuid

# ComfyUI server address (default is localhost:8188)
SERVER_ADDRESS = "127.0.0.1:8188"

def get_history(prompt_id=None, max_items=None):
    """
    Get history from ComfyUI.
    
    Args:
        prompt_id: Specific prompt ID, or None for all history
        max_items: Maximum number of items to return (for all history)
    
    Returns:
        Dictionary with prompt_id as keys, containing workflow data
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


def extract_workflow_from_history(prompt_id):
    """
    Extract workflow JSON from ComfyUI history.
    
    Args:
        prompt_id: The prompt ID from history
    
    Returns:
        Workflow JSON dict, or None if not found
    """
    history = get_history(prompt_id)
    if not history or prompt_id not in history:
        return None
    
    prompt_data = history[prompt_id]
    prompt_tuple = prompt_data.get("prompt", [])
    
    # The prompt is stored as a tuple: (number, prompt_id, prompt_json, extra_data, outputs, sensitive)
    if isinstance(prompt_tuple, (list, tuple)) and len(prompt_tuple) > 2:
        return prompt_tuple[2]  # The actual workflow is at index 2
    elif isinstance(prompt_tuple, dict):
        return prompt_tuple
    else:
        return None


def queue_prompt(prompt, prompt_id=None):
    """
    Queue a prompt to ComfyUI.
    
    Args:
        prompt: The workflow JSON (as a dict, not a string)
        prompt_id: Optional unique ID for this prompt
    """
    if prompt_id is None:
        prompt_id = str(uuid.uuid4())
    
    p = {
        "prompt": prompt,
        "client_id": str(uuid.uuid4()),  # Unique client ID
        "prompt_id": prompt_id
    }
    
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        print(f"✓ Queued prompt {prompt_id} (queue position: {result.get('number', '?')})")
        return result
    except Exception as e:
        print(f"✗ Error queueing prompt {prompt_id}: {e}")
        return None


# ============================================================================
# STEP 1: Base Workflow
# ============================================================================
# This is a fallback workflow if not loading from history or file
# You can also export from ComfyUI: File -> Export (API) and paste here

DEFAULT_WORKFLOW = """
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
            "seed": 8566257,
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

# ============================================================================
# STEP 2: Define your variations
# ============================================================================
# Each dict in this list represents one variation
# You can change any parameter that exists in your workflow

variations = [
    # Variation 1: Different seed
    {
        "name": "seed_12345",
        "changes": {
            "3": {"inputs": {"seed": 12345}}  # Node 3 (KSampler), change seed
        }
    },
    
    # Variation 2: Different CFG
    {
        "name": "cfg_5",
        "changes": {
            "3": {"inputs": {"cfg": 5}}  # Lower CFG = less prompt adherence
        }
    },
    
    # Variation 3: Different sampler
    {
        "name": "dpmpp_2m",
        "changes": {
            "3": {"inputs": {"sampler_name": "dpmpp_2m"}}
        }
    },
    
    # Variation 4: Different scheduler
    {
        "name": "karras_scheduler",
        "changes": {
            "3": {"inputs": {"scheduler": "karras"}}
        }
    },
    
    # Variation 5: More steps
    {
        "name": "30_steps",
        "changes": {
            "3": {"inputs": {"steps": 30}}
        }
    },
    
    # Variation 6: Different prompt
    {
        "name": "different_prompt",
        "changes": {
            "6": {"inputs": {"text": "masterpiece best quality man, portrait"}}
        }
    },
    
    # Variation 7: Multiple changes at once
    {
        "name": "combo_1",
        "changes": {
            "3": {
                "inputs": {
                    "seed": 99999,
                    "cfg": 7,
                    "steps": 25,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras"
                }
            }
        }
    },
    
    # Variation 8: Random seed
    {
        "name": "random_seed",
        "changes": {
            "3": {"inputs": {"seed": random.randint(0, 2**31 - 1)}}
        }
    },
    
    # Variation 9: Different image size
    {
        "name": "768x768",
        "changes": {
            "5": {"inputs": {"width": 768, "height": 768}}
        }
    },
    
    # Variation 10: Different denoise (for img2img/inpainting)
    {
        "name": "denoise_0.7",
        "changes": {
            "3": {"inputs": {"denoise": 0.7}}
        }
    },
]

# ============================================================================
# STEP 3: Generate and queue all variations
# ============================================================================

def apply_changes(base_prompt, changes):
    """
    Apply changes to a prompt dictionary.
    
    Args:
        base_prompt: The base workflow (will be copied)
        changes: Dict of changes to apply, format: {node_id: {path: value}}
    
    Example:
        changes = {
            "3": {"inputs": {"seed": 12345}}  # Change node 3's seed
        }
    """
    # Deep copy to avoid modifying the original
    prompt = json.loads(json.dumps(base_prompt))
    
    for node_id, node_changes in changes.items():
        if node_id not in prompt:
            print(f"⚠ Warning: Node {node_id} not found in workflow, skipping")
            continue
        
        # Navigate nested dict structure
        current = prompt[node_id]
        for key, value in node_changes.items():
            if key == "inputs" and isinstance(value, dict):
                # Merge inputs
                if "inputs" not in current:
                    current["inputs"] = {}
                current["inputs"].update(value)
            else:
                current[key] = value
    
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI Batch Variations Script - Create variations of a workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List recent workflows and pick one
  python3 batch_variations.py --from-history
  
  # Use a specific workflow from history
  python3 batch_variations.py --from-history abc123-def456-...
  
  # Load workflow from JSON file
  python3 batch_variations.py --workflow-file my_workflow.json
  
  # Use inline JSON (for simple workflows)
  python3 batch_variations.py --workflow-json '{"3": {...}}'
        """
    )
    parser.add_argument("--from-history", nargs="?", const="list", metavar="PROMPT_ID",
                       help="Load workflow from ComfyUI history. If no PROMPT_ID, lists recent workflows.")
    parser.add_argument("--workflow-file", type=str, metavar="FILE",
                       help="Load workflow from a JSON file")
    parser.add_argument("--workflow-json", type=str, metavar="JSON",
                       help="Load workflow from inline JSON string")
    parser.add_argument("--list-recent", type=int, default=10, metavar="N",
                       help="Number of recent workflows to show (default: 10)")
    parser.add_argument("--select", type=int, metavar="N",
                       help="Select workflow by number from --from-history list (1-based index)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ComfyUI Batch Variations Script")
    print("=" * 60)
    print(f"Server: {SERVER_ADDRESS}")
    print()
    
    # Load base workflow
    base_prompt = None
    
    if args.from_history:
        # Always fetch history first (needed for both list and select)
        print("Fetching recent workflows from ComfyUI history...")
        history = get_history(max_items=args.list_recent)
        
        if not history:
            print("✗ No history found. Run a workflow in ComfyUI first, or use --workflow-file")
            return
        
        workflow_list = list(history.items())
        
        # Handle --select option
        if args.select:
            if args.select < 1 or args.select > len(workflow_list):
                print(f"✗ Invalid selection: {args.select}. Must be between 1 and {len(workflow_list)}")
                return
            prompt_id = workflow_list[args.select - 1][0]
            print(f"Loading workflow #{args.select} from history: {prompt_id}")
            base_prompt = extract_workflow_from_history(prompt_id)
            
            if not base_prompt:
                print(f"✗ Workflow not found in history: {prompt_id}")
                return
            
            print("✓ Workflow loaded from history")
        
        elif args.from_history == "list":
            # List recent workflows
            print(f"\nFound {len(history)} recent workflow(s):\n")
            for i, (prompt_id, prompt_data) in enumerate(workflow_list, 1):
                # Try to extract some info
                prompt_tuple = prompt_data.get("prompt", [])
                sampler_info = ""
                if isinstance(prompt_tuple, (list, tuple)) and len(prompt_tuple) > 2:
                    workflow = prompt_tuple[2]
                    # Find KSampler to show some params
                    for node_id, node_data in workflow.items():
                        if node_data.get("class_type") == "KSampler":
                            inputs = node_data.get("inputs", {})
                            sampler_info = f" | seed={inputs.get('seed')}, steps={inputs.get('steps')}, cfg={inputs.get('cfg')}"
                            break
                
                # Show full prompt ID (it's usually not that long)
                print(f"  {i}. {prompt_id}{sampler_info}")
            
            print("\nTo use a workflow, you can:")
            print(f"  1. Select by number: python3 {sys.argv[0]} --from-history --select <number>")
            print(f"  2. Use full ID: python3 {sys.argv[0]} --from-history <prompt_id>")
            print("\nExample:")
            print(f"  python3 {sys.argv[0]} --from-history --select 1")
            return
        
        else:
            # Load specific workflow from history by prompt_id
            prompt_id = args.from_history
            print(f"Loading workflow from history: {prompt_id}")
            base_prompt = extract_workflow_from_history(prompt_id)
            
            if not base_prompt:
                print(f"✗ Workflow not found in history: {prompt_id}")
                print("\nTip: Use --from-history without arguments to see available workflows")
                return
            
            print("✓ Workflow loaded from history")
    
    elif args.workflow_file:
        # Load from file
        print(f"Loading workflow from file: {args.workflow_file}")
        try:
            with open(args.workflow_file, 'r') as f:
                base_prompt = json.load(f)
            print("✓ Workflow loaded from file")
        except FileNotFoundError:
            print(f"✗ File not found: {args.workflow_file}")
            return
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON file: {e}")
            return
    
    elif args.workflow_json:
        # Load from inline JSON
        print("Loading workflow from inline JSON...")
        try:
            base_prompt = json.loads(args.workflow_json)
            print("✓ Workflow loaded from JSON")
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON: {e}")
            return
    
    else:
        # Use default workflow
        print("Using default workflow (from script)")
        print("Tip: Use --from-history to load from ComfyUI, or --workflow-file to load from a file")
        try:
            base_prompt = json.loads(DEFAULT_WORKFLOW)
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing default workflow JSON: {e}")
            return
    
    if not base_prompt:
        print("✗ No workflow loaded")
        return
    
    print(f"\nVariations to queue: {len(variations)}")
    print()
    
    # Queue each variation
    import uuid
    queued = []
    
    for i, variation in enumerate(variations, 1):
        name = variation.get("name", f"variation_{i}")
        changes = variation.get("changes", {})
        
        print(f"\n[{i}/{len(variations)}] Creating variation: {name}")
        
        # Apply changes to base prompt
        modified_prompt = apply_changes(base_prompt, changes)
        
        # Generate unique prompt ID
        prompt_id = f"batch_{name}_{uuid.uuid4().hex[:8]}"
        
        # Queue it
        result = queue_prompt(modified_prompt, prompt_id)
        
        if result:
            queued.append((name, prompt_id))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Successfully queued {len(queued)}/{len(variations)} variations")
    print("\nQueued prompts:")
    for name, prompt_id in queued:
        print(f"  - {name}: {prompt_id}")
    print("\nCheck ComfyUI to see them processing!")


if __name__ == "__main__":
    import uuid  # Import here for use in functions
    main()


