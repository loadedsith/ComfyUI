#!/usr/bin/env python3
"""
Simple test script for WhisperSpeech integration.

This script tests the WhisperSpeech Pipeline directly to verify installation.
"""

import torch
from whisperspeech.pipeline import Pipeline

def test_whisperspeech():
    """Test WhisperSpeech text-to-speech generation."""
    print("Initializing WhisperSpeech Pipeline...")
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    try:
        # Initialize pipeline
        pipe = Pipeline(optimize=True, device=device)
        print("Pipeline initialized successfully!")
        
        # Test generation
        test_text = "Hello, this is a test of WhisperSpeech text-to-speech."
        print(f"\nGenerating speech for: '{test_text}'")
        
        audio = pipe.generate(
            text=test_text,
            lang='en',
            cps=15  # characters per second
        )
        
        print(f"Generated audio shape: {audio.shape}")
        print(f"Audio duration: ~{audio.shape[-1] / 24000:.2f} seconds")
        print("\n✓ WhisperSpeech is working correctly!")
        
        # Optionally save to file
        output_file = "test_output.wav"
        pipe.generate_to_file(output_file, test_text, lang='en', cps=15)
        print(f"\nSaved test audio to: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_whisperspeech()
    exit(0 if success else 1)

