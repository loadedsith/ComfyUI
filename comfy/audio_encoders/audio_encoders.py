from .wav2vec2 import Wav2Vec2Model
from .whisper import WhisperModel
import comfy.model_management
import comfy.ops
import comfy.utils
import logging
import os
import torchaudio


class AudioEncoderModel():
    def __init__(self, config):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        model_type = config.pop("model_type")
        model_config = dict(config)
        model_config.update({
            "dtype": self.dtype,
            "device": offload_device,
            "operations": comfy.ops.manual_cast
        })

        if model_type == "wav2vec2":
            self.model = Wav2Vec2Model(**model_config)
        elif model_type.startswith("whisper"):
            # Set max_chunk_length=None to allow chunking at encode_audio level
            self.model = WhisperModel(**model_config, max_chunk_length=None)
        self.model.eval()
        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
        self.model_sample_rate = 16000

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_audio(self, audio, sample_rate, max_chunk_length=None):
        """
        Encode audio with automatic chunking for long sequences.
        
        Args:
            audio: Audio tensor of shape (batch, channels, samples)
            sample_rate: Original sample rate of the audio
            max_chunk_length: Maximum chunk length in seconds. If None, uses model's max context.
        """
        comfy.model_management.load_model_gpu(self.patcher)
        audio = torchaudio.functional.resample(audio, sample_rate, self.model_sample_rate)
        audio = audio.to(self.load_device)
        
        # Get audio dimensions
        batch_size, num_channels, num_samples = audio.shape
        
        # Determine max chunk size
        if hasattr(self.model, 'get_max_audio_samples'):
            max_samples_per_chunk = self.model.get_max_audio_samples()
        else:
            # Fallback: assume 30 seconds at 16kHz
            max_samples_per_chunk = 30 * self.model_sample_rate
        
        # Override with user-specified chunk length if provided
        if max_chunk_length is not None:
            max_samples_per_chunk = int(max_chunk_length * self.model_sample_rate)
        
        # Check if chunking is needed
        if num_samples <= max_samples_per_chunk:
            # Process in one go
            out, all_layers = self.model(audio)
            outputs = {}
            outputs["encoded_audio"] = out
            outputs["encoded_audio_all_layers"] = all_layers
            outputs["audio_samples"] = num_samples
            return outputs
        
        # Chunking needed - process in non-overlapping chunks
        logging.info(f"Audio length ({num_samples/self.model_sample_rate:.2f}s) exceeds max chunk size ({max_samples_per_chunk/self.model_sample_rate:.2f}s). Processing in chunks.")
        
        chunk_outputs = []
        chunk_all_layers = []
        
        for chunk_start in range(0, num_samples, max_samples_per_chunk):
            chunk_end = min(chunk_start + max_samples_per_chunk, num_samples)
            audio_chunk = audio[:, :, chunk_start:chunk_end]
            
            # Process chunk
            out_chunk, all_layers_chunk = self.model(audio_chunk)
            chunk_outputs.append(out_chunk)
            chunk_all_layers.append(all_layers_chunk)
        
        # Concatenate chunks along sequence dimension (dim=1)
        encoded_audio = torch.cat(chunk_outputs, dim=1)
        
        # For all_layers, we need to concatenate each layer's outputs
        # all_layers is a tuple where each element is a tensor of shape (batch, seq, features)
        num_layers = len(chunk_all_layers[0])
        concatenated_layers = []
        for layer_idx in range(num_layers):
            layer_chunks = [chunk_layers[layer_idx] for chunk_layers in chunk_all_layers]
            concatenated_layer = torch.cat(layer_chunks, dim=1)
            concatenated_layers.append(concatenated_layer)
        
        outputs = {}
        outputs["encoded_audio"] = encoded_audio
        outputs["encoded_audio_all_layers"] = tuple(concatenated_layers)
        outputs["audio_samples"] = num_samples
        return outputs


def audio_encoder_detection_error_hint(path, state_dict):
    """
    Provides helpful hints when audio encoder detection fails.
    Similar to model_detection_error_hint in comfy/sd.py
    """
    filename = os.path.basename(path) if path else "unknown"
    hints = []
    
    # Check for common key patterns
    has_wav2vec2_keys = any("encoder.layer_norm" in k for k in state_dict.keys())
    has_whisper_keys = any("encoder.embed_positions" in k or "model.encoder.embed_positions" in k for k in state_dict.keys())
    
    if has_wav2vec2_keys:
        hints.append("This appears to be a wav2vec2 model.")
    elif has_whisper_keys:
        hints.append("This appears to be a Whisper model.")
    
    # Check for common file naming patterns
    if 'whisper' in filename.lower():
        hints.append("Filename suggests this is a Whisper model.")
    elif 'wav2vec' in filename.lower():
        hints.append("Filename suggests this is a wav2vec2 model.")
    
    if hints:
        return "\nHINT: " + " ".join(hints)
    return ""


def detect_whisper_model_size(state_dict):
    """
    Detect Whisper model size from state dict dimensions.
    
    Returns tuple: (n_audio_state, n_audio_head, n_audio_layer, model_name)
    """
    # Try to find encoder embedding dimension from various possible keys
    embed_key = None
    for key in ["encoder.embed_positions.weight", "model.encoder.embed_positions.weight"]:
        if key in state_dict:
            embed_key = key
            break
    
    if embed_key is None:
        return None
    
    embed_dim = state_dict[embed_key].shape[1] if len(state_dict[embed_key].shape) > 1 else state_dict[embed_key].shape[0]
    
    # Whisper model size configurations
    # Format: (n_audio_state, n_audio_head, n_audio_layer, model_name)
    whisper_configs = {
        384: (384, 6, 4, "tiny"),
        512: (512, 8, 6, "base"),
        768: (768, 12, 12, "small"),
        1024: (1024, 16, 24, "medium"),
        1280: (1280, 20, 32, "large-v3"),  # large-v2 and large-v3 have same architecture
    }
    
    if embed_dim in whisper_configs:
        return whisper_configs[embed_dim]
    
    # If dimension doesn't match known sizes, try to infer from layer count
    # Count encoder layers
    layer_keys = [k for k in state_dict.keys() if "encoder.layers" in k and "weight" in k]
    if layer_keys:
        # Extract layer indices
        layer_indices = set()
        for key in layer_keys:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_indices.add(layer_idx)
                    except ValueError:
                        pass
        
        if layer_indices:
            max_layer = max(layer_indices)
            # Estimate based on layer count
            if max_layer < 5:
                return (384, 6, 4, "tiny")
            elif max_layer < 7:
                return (512, 8, 6, "base")
            elif max_layer < 13:
                return (768, 12, 12, "small")
            elif max_layer < 25:
                return (1024, 16, 24, "medium")
            else:
                return (1280, 20, 32, "large-v3")
    
    return None


def load_audio_encoder_from_sd(sd, prefix=""):
    """
    Load audio encoder from state dict with automatic model type detection.
    
    Supports:
    - wav2vec2 models (base and large variants)
    - Whisper models (tiny, base, small, medium, large-v2, large-v3)
    
    Models should be PyTorch state dicts (.pt, .safetensors, etc.) placed in
    models/audio_encoders/ directory.
    """
    original_sd = sd.copy()
    sd = comfy.utils.state_dict_prefix_replace(sd, {"wav2vec2.": ""})
    
    # Detect wav2vec2 models
    if "encoder.layer_norm.bias" in sd:
        embed_dim = sd["encoder.layer_norm.bias"].shape[0]
        if embed_dim == 1024:  # large
            config = {
                "model_type": "wav2vec2",
                "embed_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "conv_norm": True,
                "conv_bias": True,
                "do_normalize": True,
                "do_stable_layer_norm": True
            }
        elif embed_dim == 768:  # base
            config = {
                "model_type": "wav2vec2",
                "embed_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "conv_norm": False,
                "conv_bias": False,
                "do_normalize": False,  # chinese-wav2vec2-base has this False
                "do_stable_layer_norm": False
            }
        else:
            raise RuntimeError(
                "ERROR: audio encoder file is invalid or unsupported wav2vec2 embed_dim: {}. "
                "Supported dimensions are 768 (base) and 1024 (large).".format(embed_dim)
            )
    
    # Detect Whisper models
    elif "model.encoder.embed_positions.weight" in sd or "encoder.embed_positions.weight" in sd:
        # Normalize state dict keys (remove "model." prefix if present)
        # Do this before detection since detect_whisper_model_size expects normalized keys
        sd = comfy.utils.state_dict_prefix_replace(sd, {"model.": ""})
        
        # Detect model size (after normalization)
        whisper_config = detect_whisper_model_size(sd)
        if whisper_config is None:
            # Fallback to large-v3 if detection fails (for backward compatibility)
            logging.warning(
                "Could not detect Whisper model size from state dict dimensions. "
                "Defaulting to large-v3 configuration. "
                "If this model is a different size, please ensure the state dict contains "
                "encoder.embed_positions.weight with correct dimensions."
            )
            whisper_config = (1280, 20, 32, "large-v3")
        
        n_audio_state, n_audio_head, n_audio_layer, model_name = whisper_config
        config = {
            "model_type": f"whisper-{model_name}",
            "n_audio_state": n_audio_state,
            "n_audio_head": n_audio_head,
            "n_audio_layer": n_audio_layer,
        }
        logging.info(f"Detected Whisper model: {model_name} (n_audio_state={n_audio_state}, n_audio_head={n_audio_head}, n_audio_layer={n_audio_layer})")
    
    else:
        # Try to provide helpful error message
        sample_keys = list(original_sd.keys())[:10]
        error_msg = (
            "ERROR: audio encoder not supported. Could not detect model type from state dict.\n"
            f"Sample keys found: {sample_keys[:5]}...\n"
            "Expected keys for wav2vec2: 'encoder.layer_norm.bias'\n"
            "Expected keys for Whisper: 'encoder.embed_positions.weight' or 'model.encoder.embed_positions.weight'"
        )
        raise RuntimeError(error_msg)

    audio_encoder = AudioEncoderModel(config)
    m, u = audio_encoder.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing audio encoder keys: {}".format(m))
    if len(u) > 0:
        logging.warning("unexpected audio encoder keys: {}".format(u))

    return audio_encoder
