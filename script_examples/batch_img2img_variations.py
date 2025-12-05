#!/usr/bin/env python3
"""
ComfyUI Batch Variations Script - Customized for Image-to-Image Workflow

This script is customized for your "Simple Image To Image" workflow.
It will queue multiple variations with different settings.

HOW TO USE:
1. Make sure ComfyUI is running (python3 main.py)
2. Run: python3 script_examples/batch_img2img_variations.py
"""

import json
import urllib.request
import random
import uuid

# ComfyUI server address
SERVER_ADDRESS = "127.0.0.1:8188"

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
        print(f"✓ Queued: {prompt_id[:16]}... (queue #{result.get('number', '?')})")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# Your workflow (from Simple Image To Image.json)
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

# ============================================================================
# Define your variations
# ============================================================================
# Node IDs in your workflow:
#   3 = KSampler (seed, steps, cfg, sampler_name, scheduler, denoise)
#   6 = Positive prompt (CLIPTextEncode)
#   7 = Negative prompt (CLIPTextEncode)
#   9 = SaveImage (filename_prefix)
#   19 = LoadImage (image file)

variations = [
    # Variation 1: Different seeds (10 random seeds)
    *[{
        "name": f"seed_{i+1}",
        "changes": {
            "3": {"inputs": {"seed": random.randint(0, 2**31 - 1)}}
        }
    } for i in range(10)],
    
    # Variation 11: Different denoise values
    {
        "name": "denoise_0.3",
        "changes": {
            "3": {"inputs": {"denoise": 0.3}}  # More similar to original
        }
    },
    {
        "name": "denoise_0.7",
        "changes": {
            "3": {"inputs": {"denoise": 0.7}}  # More different from original
        }
    },
    {
        "name": "denoise_0.9",
        "changes": {
            "3": {"inputs": {"denoise": 0.9}}  # Very different
        }
    },
    
    # Variation 14: Different CFG values
    {
        "name": "cfg_7",
        "changes": {
            "3": {"inputs": {"cfg": 7}}  # Less prompt adherence
        }
    },
    {
        "name": "cfg_15",
        "changes": {
            "3": {"inputs": {"cfg": 15}}  # More prompt adherence
        }
    },
    
    # Variation 16: Different samplers
    {
        "name": "sampler_dpmpp_2m",
        "changes": {
            "3": {"inputs": {"sampler_name": "dpmpp_2m"}}
        }
    },
    {
        "name": "sampler_euler",
        "changes": {
            "3": {"inputs": {"sampler_name": "euler"}}
        }
    },
    {
        "name": "sampler_dpmpp_2m_sde",
        "changes": {
            "3": {"inputs": {"sampler_name": "dpmpp_2m_sde"}}
        }
    },
    
    # Variation 19: Different schedulers
    {
        "name": "scheduler_normal",
        "changes": {
            "3": {"inputs": {"scheduler": "normal"}}
        }
    },
    {
        "name": "scheduler_exponential",
        "changes": {
            "3": {"inputs": {"scheduler": "exponential"}}
        }
    },
    
    # Variation 21: Different step counts
    {
        "name": "steps_50",
        "changes": {
            "3": {"inputs": {"steps": 50}}  # Faster, lower quality
        }
    },
    {
        "name": "steps_150",
        "changes": {
            "3": {"inputs": {"steps": 150}}  # Slower, higher quality
        }
    },
    
    # Variation 23: Different prompts
    {
        "name": "prompt_warrior",
        "changes": {
            "6": {"inputs": {"text": "Mortal Kombat warrior, fierce fighter, detailed armor\n"}}
        }
    },
    {
        "name": "prompt_ninja",
        "changes": {
            "6": {"inputs": {"text": "Mortal Kombat ninja, stealthy assassin, dark outfit\n"}}
        }
    },
    
    # Variation 25: Combined changes (multiple parameters at once)
    {
        "name": "combo_fast",
        "changes": {
            "3": {
                "inputs": {
                    "seed": random.randint(0, 2**31 - 1),
                    "steps": 50,
                    "cfg": 8,
                    "sampler_name": "dpmpp_2m",
                    "denoise": 0.6
                }
            }
        }
    },
    {
        "name": "combo_quality",
        "changes": {
            "3": {
                "inputs": {
                    "seed": random.randint(0, 2**31 - 1),
                    "steps": 150,
                    "cfg": 12,
                    "sampler_name": "dpmpp_2m_sde",
                    "denoise": 0.7
                }
            }
        }
    },
]


def apply_changes(base_prompt, changes):
    """Apply changes to a prompt dictionary."""
    prompt = json.loads(json.dumps(base_prompt))  # Deep copy
    
    for node_id, node_changes in changes.items():
        if node_id not in prompt:
            print(f"⚠ Warning: Node {node_id} not found, skipping")
            continue
        
        current = prompt[node_id]
        for key, value in node_changes.items():
            if key == "inputs" and isinstance(value, dict):
                if "inputs" not in current:
                    current["inputs"] = {}
                current["inputs"].update(value)
            else:
                current[key] = value
    
    return prompt


def main():
    print("=" * 70)
    print("ComfyUI Image-to-Image Batch Variations")
    print("=" * 70)
    print(f"Server: {SERVER_ADDRESS}")
    print(f"Total variations: {len(variations)}")
    print()
    
    # Parse workflow
    try:
        base_prompt = json.loads(workflow_json)
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing workflow JSON: {e}")
        return
    
    # Queue all variations
    queued = []
    for i, variation in enumerate(variations, 1):
        name = variation.get("name", f"var_{i}")
        changes = variation.get("changes", {})
        
        print(f"[{i:2d}/{len(variations)}] {name}")
        
        modified_prompt = apply_changes(base_prompt, changes)
        prompt_id = f"img2img_{name}_{uuid.uuid4().hex[:8]}"
        
        result = queue_prompt(modified_prompt, prompt_id)
        if result:
            queued.append((name, prompt_id))
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✓ Successfully queued {len(queued)}/{len(variations)} variations")
    print("=" * 70)
    print("\nCheck ComfyUI to see them processing!")
    print("Images will be saved with prefix 'SD1.5' in the output folder.")


if __name__ == "__main__":
    main()



