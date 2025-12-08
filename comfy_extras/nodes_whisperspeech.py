"""
WhisperSpeech TTS node with support for long text via chunking.

Handles text-to-speech generation with automatic chunking when text exceeds
the model's context window. Finds word boundaries to ensure natural breaks.
"""

import torch
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

import folder_paths

def get_whisperspeech_models():
    """
    Get list of available WhisperSpeech models from local models/audio_encoders folder.
    Returns tuple of (t2s_models, s2a_models) with filenames.
    """
    try:
        # Get models from audio_encoders folder (filter for .model files)
        all_models = folder_paths.get_filename_list("audio_encoders")
        local_models = [m for m in all_models if m.endswith('.model')]
        
        t2s_models = []
        s2a_models = []
        
        for model in sorted(local_models):
            if model.startswith('t2s-') and model.endswith('.model'):
                t2s_models.append(model)
            elif model.startswith('s2a-') and model.endswith('.model'):
                s2a_models.append(model)
        
        # Add "default" option at the beginning (use empty string for "None")
        t2s_models = [""] + t2s_models
        s2a_models = [""] + s2a_models
        
        return (t2s_models, s2a_models)
    except Exception as e:
        logging.warning(f"Could not list WhisperSpeech models from folder: {e}. Using empty list.")
        # Return empty lists with just default option
        return ([""], [""])


def find_word_boundaries(text: str) -> List[int]:
    """
    Find word boundary positions in text.
    Returns list of character indices where words start (including start of text).
    """
    boundaries = [0]  # Start of text is always a boundary
    # Find word boundaries using regex
    for match in re.finditer(r'\b', text):
        pos = match.start()
        if pos > 0 and pos < len(text):
            boundaries.append(pos)
    boundaries.append(len(text))  # End of text
    return sorted(set(boundaries))


def estimate_text_position(audio_duration: float, cps: float, total_chars: int) -> int:
    """
    Estimate character position in text based on audio duration and characters per second.
    
    Args:
        audio_duration: Duration of generated audio in seconds
        cps: Characters per second (speaking rate)
        total_chars: Total characters in text
    
    Returns:
        Estimated character position (clamped to valid range)
    """
    estimated_chars = int(audio_duration * cps)
    return min(estimated_chars, total_chars)


def find_nearest_word_boundary(text: str, target_pos: int, boundaries: List[int], lookback_chars: int = 50) -> int:
    """
    Find the nearest word boundary before or at target position.
    
    Args:
        text: Full text
        target_pos: Target character position
        boundaries: List of word boundary positions
        lookback_chars: Maximum characters to look back for a boundary
    
    Returns:
        Character position of nearest word boundary
    """
    # Clamp target position
    target_pos = max(0, min(target_pos, len(text)))
    
    # Find boundaries within lookback range
    start_pos = max(0, target_pos - lookback_chars)
    valid_boundaries = [b for b in boundaries if start_pos <= b <= target_pos]
    
    if valid_boundaries:
        # Return the boundary closest to target (prefer earlier boundaries)
        return max(valid_boundaries)
    
    # Fallback: return start_pos or 0
    return max(0, start_pos)


def chunk_text_for_tts(
    text: str,
    max_chunk_duration: float,
    cps: float,
    lookback_chars: int = 50
) -> List[Tuple[int, int, str]]:
    """
    Split text into chunks suitable for TTS generation.
    
    Uses word boundaries to ensure natural breaks. Estimates chunk size based on
    characters per second, but finds the nearest word boundary to avoid cutting
    words in half.
    
    Args:
        text: Full text to chunk
        max_chunk_duration: Maximum duration per chunk in seconds
        cps: Characters per second (speaking rate)
        lookback_chars: Maximum characters to look back for word boundary
    
    Returns:
        List of tuples: (start_pos, end_pos, chunk_text)
    """
    if not text.strip():
        return []
    
    boundaries = find_word_boundaries(text)
    chunks = []
    current_pos = 0
    total_chars = len(text)
    
    # Calculate max characters per chunk (with some margin for safety)
    max_chars_per_chunk = int(max_chunk_duration * cps * 0.9)  # 90% to be safe
    
    # Minimum chunk size to ensure progress
    min_chunk_size = max(10, max_chars_per_chunk // 4)
    
    while current_pos < total_chars:
        remaining_chars = total_chars - current_pos
        
        # If remaining text is small enough (less than 1.2x max chunk), take it all
        if remaining_chars <= max_chars_per_chunk * 1.2:
            chunk_end = total_chars
        else:
            # Estimate end position for this chunk
            target_end = min(current_pos + max_chars_per_chunk, total_chars)
            
            # Find nearest word boundary before target
            chunk_end = find_nearest_word_boundary(text, target_end, boundaries, lookback_chars)
            
            # Ensure we make progress (don't get stuck at same position)
            if chunk_end <= current_pos:
                # No boundary found, take max_chars_per_chunk or find next boundary
                chunk_end = min(current_pos + max_chars_per_chunk, total_chars)
                # Try to find any boundary after current_pos
                next_boundaries = [b for b in boundaries if b > current_pos and b <= chunk_end]
                if next_boundaries:
                    chunk_end = next_boundaries[0]
                # If still no boundary found, force a chunk of max size
                if chunk_end == current_pos:
                    chunk_end = min(current_pos + max_chars_per_chunk, total_chars)
            
            # Ensure minimum chunk size (unless it's truly the last bit of text)
            if chunk_end - current_pos < min_chunk_size and remaining_chars > max_chars_per_chunk:
                # Look for a boundary further ahead
                further_boundaries = [b for b in boundaries if b > chunk_end and b <= current_pos + max_chars_per_chunk * 1.5]
                if further_boundaries:
                    chunk_end = further_boundaries[0]
                # If no further boundary, take max_chars_per_chunk
                elif chunk_end - current_pos < min_chunk_size:
                    chunk_end = min(current_pos + max_chars_per_chunk, total_chars)
        
        # Extract chunk text
        chunk_text = text[current_pos:chunk_end].strip()
        
        # Always add chunk if it has content
        if chunk_text:
            chunks.append((current_pos, chunk_end, chunk_text))
        
        # Always update position to make progress
        # If chunk_end equals current_pos, we need to force progress
        if chunk_end > current_pos:
            current_pos = chunk_end
        else:
            # Stuck - force progress by taking at least one character
            current_pos = min(current_pos + 1, total_chars)
            if current_pos < total_chars:
                # Try to find next word boundary
                next_boundaries = [b for b in boundaries if b > current_pos]
                if next_boundaries:
                    current_pos = next_boundaries[0]
    
    # Final check: ensure we processed everything up to the end
    # This is critical - always add remaining text even if it's small
    if current_pos < total_chars:
        remaining = text[current_pos:]
        if remaining.strip():  # Has non-whitespace content
            logging.info(f"Adding final remaining text: {len(remaining.strip())} chars (pos {current_pos} to {total_chars})")
            chunks.append((current_pos, total_chars, remaining.strip()))
        else:
            # Even if it's just whitespace, include it to ensure we process everything
            logging.info(f"Adding final whitespace-only text (pos {current_pos} to {total_chars})")
            chunks.append((current_pos, total_chars, remaining.strip()))
    
    return chunks


def apply_fade_out(audio_tensor, fade_duration_ms=20, sample_rate=24000):
    """
    Apply a fade-out to the end of audio to prevent clicks.
    
    Args:
        audio_tensor: Audio tensor of shape [batch, channels, samples]
        fade_duration_ms: Fade-out duration in milliseconds
        sample_rate: Sample rate of the audio
    
    Returns:
        Audio tensor with fade-out applied
    """
    fade_samples = int(fade_duration_ms * sample_rate / 1000.0)
    batch_size, num_channels, num_samples = audio_tensor.shape
    
    if num_samples <= fade_samples:
        # Audio is shorter than fade duration, fade entire audio
        fade_samples = num_samples
    
    # Create fade-out envelope (linear fade from 1.0 to 0.0)
    fade_envelope = torch.linspace(1.0, 0.0, fade_samples, device=audio_tensor.device, dtype=audio_tensor.dtype)
    
    # Apply fade to last fade_samples of each channel
    audio_tensor = audio_tensor.clone()  # Don't modify original
    audio_tensor[:, :, -fade_samples:] *= fade_envelope.unsqueeze(0).unsqueeze(0)
    
    return audio_tensor


class WhisperSpeechGenerate(io.ComfyNode):
    """
    Generate speech from text using WhisperSpeech with sliding window chunking.
    
    For long text, automatically splits into chunks at word boundaries. Uses a sliding
    window approach with token prompts (stoks_prompt and atoks_prompt) to maintain
    smooth transitions between chunks, preserving prosody, intonation, and tonal
    continuity.
    
    The sliding window works by:
    1. Generating first chunk normally
    2. Extracting last N tokens (semantic + audio) from generated chunk
    3. Using those tokens as prompts for next chunk generation
    4. This maintains context and prosody across chunk boundaries
    
    Note: Chunk boundaries are estimated based on measured speaking rate since
    WhisperSpeech doesn't provide text position feedback during generation.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WhisperSpeechGenerate",
            category="audio",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip="Text to convert to speech. Long text will be automatically chunked at word boundaries."
                ),
                io.String.Input(
                    "lang",
                    default="en",
                    tooltip="Language code (e.g., 'en', 'es', 'fr'). Note: Only base language codes are supported (not regional variants like 'en-US' or 'en-GB'). Regional variants will be automatically converted to base codes. Speaker embeddings capture voice characteristics (timbre, pitch) but not accent. Accent comes from the model's training data bias."
                ),
                io.Audio.Input(
                    "reference_audio",
                    optional=True,
                    tooltip="Optional reference audio for voice cloning. Extracts speaker embedding to match voice characteristics (timbre, pitch, tone). All WhisperSpeech models support this feature. Note: Accent/pronunciation comes from the model's training data, not the speaker embedding. Longer reference audio (10-30s) with clear speech works best."
                ),
                io.Float.Input(
                    "cps",
                    default=15.0,
                    min=5.0,
                    max=50.0,
                    tooltip="Characters per second (speaking rate). Used to estimate chunk boundaries."
                ),
                io.Float.Input(
                    "max_chunk_duration",
                    default=25.0,
                    min=5.0,
                    max=30.0,
                    tooltip="Maximum duration per chunk in seconds. Text exceeding this will be split at word boundaries."
                ),
                io.Int.Input(
                    "lookback_chars",
                    default=50,
                    min=0,
                    max=200,
                    tooltip="Maximum characters to look back when finding word boundaries for chunking."
                ),
                io.Int.Input(
                    "prompt_tokens",
                    default=50,
                    min=10,
                    max=200,
                    tooltip="Number of tokens from previous chunk to use as prompt for continuity. "
                           "Higher values maintain better prosody/intonation but use more memory."
                ),
                io.Combo.Input(
                    "t2s_ref",
                    options=get_whisperspeech_models()[0],
                    default="",
                    optional=True,
                    tooltip="Text-to-semantic model from models/audio_encoders folder. Empty string uses default. Download models using script_examples/download_whisperspeech_models.py"
                ),
                io.Combo.Input(
                    "s2a_ref",
                    options=get_whisperspeech_models()[1],
                    default="",
                    optional=True,
                    tooltip="Semantic-to-audio model from models/audio_encoders folder. Empty string uses default. Download models using script_examples/download_whisperspeech_models.py"
                ),
            ],
            outputs=[io.Audio.Output()],
        )
    
    @classmethod
    def execute(
        cls,
        text: str,
        lang: str = "en",
        cps: float = 15.0,
        max_chunk_duration: float = 25.0,
        lookback_chars: int = 50,
        prompt_tokens: int = 50,
        reference_audio: Optional[io.Audio] = None,
        t2s_ref: Optional[str] = None,
        s2a_ref: Optional[str] = None
    ) -> io.NodeOutput:
        # Normalize language code - WhisperSpeech doesn't support regional variants (e.g., en-US, en-GB)
        # Strip regional suffix if present (e.g., "en-US" -> "en", "es-MX" -> "es")
        if '-' in lang or '_' in lang:
            base_lang = lang.split('-')[0].split('_')[0]
            if base_lang != lang:
                logging.warning(f"Language code '{lang}' contains regional variant. WhisperSpeech only supports base language codes. Using '{base_lang}' instead.")
            lang = base_lang
        try:
            from whisperspeech.pipeline import Pipeline
        except ImportError:
            raise RuntimeError(
                "WhisperSpeech is not installed. Install it with: pip install whisperspeech"
            )
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize pipeline with optional custom model references
        # t2s_ref: text-to-semantic model (controls speed, quality, languages)
        # s2a_ref: semantic-to-audio model (controls speed, quality)
        # Empty string means use default model
        # If a filename is provided, resolve it to full path from models/whisperspeech folder
        t2s_model_ref = None
        s2a_model_ref = None
        
        if t2s_ref and t2s_ref.strip():
            # Check if it's a local file (just filename) or HuggingFace reference (contains ':')
            if ':' in t2s_ref:
                # HuggingFace reference format: "collabora/whisperspeech:filename.model"
                t2s_model_ref = t2s_ref
                logging.info(f"Using HuggingFace T2S model: {t2s_model_ref}")
            else:
                # Local file from models/audio_encoders folder
                try:
                    t2s_model_ref = folder_paths.get_full_path_or_raise("audio_encoders", t2s_ref)
                    logging.info(f"Using local T2S model: {t2s_model_ref}")
                except Exception as e:
                    logging.warning(f"Could not find local T2S model '{t2s_ref}': {e}. Using default.")
                    t2s_model_ref = None
        
        if s2a_ref and s2a_ref.strip():
            # Check if it's a local file (just filename) or HuggingFace reference (contains ':')
            if ':' in s2a_ref:
                # HuggingFace reference format: "collabora/whisperspeech:filename.model"
                s2a_model_ref = s2a_ref
                logging.info(f"Using HuggingFace S2A model: {s2a_model_ref}")
            else:
                # Local file from models/audio_encoders folder
                try:
                    s2a_model_ref = folder_paths.get_full_path_or_raise("audio_encoders", s2a_ref)
                    logging.info(f"Using local S2A model: {s2a_model_ref}")
                except Exception as e:
                    logging.warning(f"Could not find local S2A model '{s2a_ref}': {e}. Using default.")
                    s2a_model_ref = None
        
        pipe = Pipeline(
            t2s_ref=t2s_model_ref,
            s2a_ref=s2a_model_ref,
            optimize=True,
            device=device
        )
        
        sample_rate = 24000  # WhisperSpeech default
        
        def normalize_audio_shape(audio_tensor):
            """Normalize audio tensor to [1, channels, samples] format."""
            if len(audio_tensor.shape) == 1:
                # [samples] -> [1, 1, samples]
                return audio_tensor.unsqueeze(0).unsqueeze(0)
            elif len(audio_tensor.shape) == 2:
                # [channels, samples] or [samples, channels]
                if audio_tensor.shape[0] > audio_tensor.shape[1]:
                    # [samples, channels] -> [channels, samples]
                    audio_tensor = audio_tensor.transpose(0, 1)
                # [channels, samples] -> [1, channels, samples]
                return audio_tensor.unsqueeze(0)
            elif len(audio_tensor.shape) == 3:
                # Already [batch, channels, samples] or [batch, samples, channels]
                if audio_tensor.shape[0] != 1:
                    audio_tensor = audio_tensor[0:1]  # Take first batch
                # Check if we need to transpose
                if audio_tensor.shape[1] > audio_tensor.shape[2] and audio_tensor.shape[2] < 10:
                    # Likely [batch, samples, channels], transpose
                    audio_tensor = audio_tensor.transpose(1, 2)
                return audio_tensor
            else:
                raise ValueError(f"Unexpected audio tensor shape: {audio_tensor.shape}")
        
        # WhisperSpeech doesn't provide duration limits or text position feedback.
        # step_callback is called during generation but provides no arguments.
        # Strategy: Use step_callback to track progress and implement duration-based early exit.
        
        # First, try generating the full text - WhisperSpeech might handle it
        estimated_duration = len(text) / cps if cps > 0 else 0
        
        if estimated_duration <= max_chunk_duration:
            # Try generating in one go with step tracking
            logging.info(f"Attempting to generate full text ({len(text)} chars, estimated ~{estimated_duration:.1f}s)")
            try:
                # Use step_callback to track generation progress
                step_info = {'s2a_steps': 0}  # Track s2a steps (correlate to audio duration)
                
                def track_progress():
                    step_info['s2a_steps'] += 1
                
                audio = pipe.generate(text=text, lang=lang, cps=cps, step_callback=track_progress)
                audio = normalize_audio_shape(audio)
                
                actual_duration = audio.shape[2] / sample_rate if len(audio.shape) >= 3 else audio.shape[1] / sample_rate
                logging.info(f"Generated full text successfully ({actual_duration:.2f}s, {step_info['s2a_steps']} s2a steps)")
                
                return io.NodeOutput({
                    "waveform": audio,
                    "sample_rate": sample_rate
                })
            except Exception as e:
                # If generation fails, fall back to chunking
                logging.warning(f"Full text generation failed, falling back to chunking: {e}")
        
        # Chunking needed - use step_callback to measure actual generation rate
        logging.info(f"Text appears long ({len(text)} chars). Using step_callback to measure generation rate...")
        
        # Measure actual CPS by generating a sample with step tracking
        sample_size = min(100, len(text))
        step_info = {'s2a_steps': 0}
        
        def track_steps():
            step_info['s2a_steps'] += 1
        
        try:
            test_audio = pipe.generate(text=text[:sample_size], lang=lang, cps=cps, step_callback=track_steps)
            test_audio = normalize_audio_shape(test_audio)
            test_duration = test_audio.shape[2] / sample_rate if len(test_audio.shape) >= 3 else test_audio.shape[1] / sample_rate
            
            # Calculate actual CPS from measured duration
            actual_cps = sample_size / test_duration if test_duration > 0 else cps
            
            # Also calculate steps per second for future duration estimation
            steps_per_second = step_info['s2a_steps'] / test_duration if test_duration > 0 else None
            
            logging.info(
                f"Measured: {actual_cps:.2f} chars/sec, "
                f"{steps_per_second:.1f} s2a steps/sec (user CPS estimate: {cps:.2f})"
            )
        except Exception as e:
            logging.warning(f"Could not measure generation rate, using user estimate: {e}")
            actual_cps = cps
            steps_per_second = None
        
        # Re-estimate duration with measured CPS
        estimated_duration = len(text) / actual_cps if actual_cps > 0 else 0
        
        logging.info(
            f"Text length ({len(text)} chars) estimated at ~{estimated_duration:.1f}s. "
            f"Splitting into chunks at word boundaries (max {max_chunk_duration}s per chunk)."
        )
        
        # Split text into chunks using measured CPS
        chunks = chunk_text_for_tts(text, max_chunk_duration, actual_cps, lookback_chars)
        
        # Verify all text is covered
        total_chunked_chars = sum(end - start for start, end, _ in chunks)
        if total_chunked_chars < len(text):
            missing = len(text) - total_chunked_chars
            logging.warning(f"Chunking missed {missing} characters! Total text: {len(text)}, chunked: {total_chunked_chars}")
            # Add the missing text as a final chunk
            last_end = chunks[-1][1] if chunks else 0
            if last_end < len(text):
                missing_text = text[last_end:].strip()
                if missing_text:
                    logging.info(f"Adding missing text as final chunk: '{missing_text[:50]}...'")
                    chunks.append((last_end, len(text), missing_text))
        
        logging.info(f"Split text into {len(chunks)} chunks (total {sum(end - start for start, end, _ in chunks)} chars)")
        
        # Generate audio for each chunk with sliding window continuity
        # Use atoks_prompt to maintain prosody/intonation across chunks
        # Note: stoks_prompt is NOT used because it causes text repetition
        # (t2s.generate prepends stoks_prompt, duplicating tokens in the output)
        audio_chunks = []
        stoks_prompt = None  # Not used - causes text repetition
        atoks_prompt = None  # Audio tokens from previous chunk (used for continuity)
        
        # Get speaker embedding
        # If reference_audio is provided, extract speaker embedding from it (voice cloning)
        # Otherwise, use default speaker
        # Check if reference_audio is valid (not None, is dict, has required keys, and has actual data)
        reference_audio_valid = (
            reference_audio is not None and 
            isinstance(reference_audio, dict) and 
            'waveform' in reference_audio and 
            'sample_rate' in reference_audio and
            reference_audio['waveform'] is not None and
            isinstance(reference_audio['waveform'], torch.Tensor) and
            reference_audio['waveform'].numel() > 0  # Has actual data
        )
        
        if reference_audio_valid:
            # Extract speaker embedding from reference audio
            # ComfyUI Audio format: {"waveform": [B, C, T], "sample_rate": int}
            import tempfile
            import torchaudio
            import os
            
            ref_waveform = reference_audio['waveform']  # [B, C, T]
            ref_sample_rate = reference_audio['sample_rate']
            
            # Validate waveform has data
            if ref_waveform.numel() == 0:
                raise ValueError("Reference audio waveform is empty (zero elements)")
            
            # Normalize to [C, T] format for torchaudio.save
            # torchaudio.save expects 2D tensor [C, T], not 3D [B, C, T]
            
            # Take first batch if multiple batches, or remove batch dimension
            if len(ref_waveform.shape) == 0:
                raise ValueError(f"Reference audio waveform has invalid shape: {ref_waveform.shape}")
            elif ref_waveform.shape[0] > 1:
                ref_waveform = ref_waveform[0]  # [B, C, T] -> [C, T]
            elif ref_waveform.shape[0] == 0:
                raise ValueError("Reference audio waveform has zero batch size")
            else:
                # Remove batch dimension: [1, C, T] -> [C, T]
                # Use indexing instead of squeeze to avoid removing channel dim if C=1
                ref_waveform = ref_waveform[0]  # [1, C, T] -> [C, T]
            
            # Take first channel if multi-channel (mono is sufficient for speaker embedding)
            # After removing batch, ref_waveform is [C, T]
            if len(ref_waveform.shape) == 2 and ref_waveform.shape[0] > 1:
                ref_waveform = ref_waveform[0:1]  # [C, T] -> [1, T] for mono
            
            # Ensure waveform is in correct format [C, T] (2D)
            if len(ref_waveform.shape) != 2:
                raise ValueError(f"Expected waveform shape [C, T] after normalization, got {ref_waveform.shape}")
            
            # Save to temporary file for extract_spk_emb
            # extract_spk_emb expects a file path
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
                # Save audio file (torchaudio.save expects [C, T] - 2D tensor)
                torchaudio.save(tmp_path, ref_waveform, ref_sample_rate)
                
                try:
                    # Extract speaker embedding from reference audio
                    # ref_waveform is now [C, T], so time dimension is shape[1]
                    duration = ref_waveform.shape[1] / ref_sample_rate
                    logging.info(f"Extracting speaker embedding from reference audio (duration: {duration:.2f}s, sample_rate: {ref_sample_rate}Hz)...")
                    
                    # Verify pipeline has extract_spk_emb method (all models should support this)
                    if not hasattr(pipe, 'extract_spk_emb'):
                        logging.warning(f"Pipeline does not have extract_spk_emb method. This model may not support voice cloning. Using default speaker.")
                        speaker = pipe.default_speaker
                    else:
                        try:
                            speaker = pipe.extract_spk_emb(tmp_path)
                            logging.info(f"Successfully extracted speaker embedding: shape {speaker.shape}")
                        except Exception as e:
                            logging.warning(f"Failed to extract speaker embedding from reference audio: {e}. Using default speaker.")
                            import traceback
                            logging.debug(f"Traceback: {traceback.format_exc()}")
                            speaker = pipe.default_speaker
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        else:
            # Use default speaker
            speaker = pipe.default_speaker
            if reference_audio is not None:
                logging.warning(f"Reference audio provided but invalid. Type: {type(reference_audio)}, "
                              f"Keys: {list(reference_audio.keys()) if isinstance(reference_audio, dict) else 'N/A'}. "
                              f"Using default speaker embedding.")
            else:
                logging.info("No reference audio provided, using default speaker embedding")
        
        # Normalize speaker embedding once before the loop (reuse for all chunks)
        # s2a.generate() expects speaker as 2D [batch, features]
        original_speaker_shape = speaker.shape
        if len(speaker.shape) == 1:
            # [features] -> [1, features]
            speaker_normalized = speaker.unsqueeze(0)
        elif len(speaker.shape) == 2:
            # Already 2D [batch, features] - ensure batch dim is 1
            if speaker.shape[0] > 1:
                speaker_normalized = speaker[0:1].clone()  # Take first batch only, clone to avoid modifying original
            else:
                speaker_normalized = speaker.clone()  # Clone to avoid modifying original
        else:
            # 3D+ - flatten to 2D
            logging.warning(f"  Unexpected speaker shape {speaker.shape}, flattening to 2D")
            # Flatten all but last dimension
            if len(speaker.shape) > 2:
                batch_size = speaker.shape[0]
                feature_size = speaker.shape[-1]
                speaker_flat = speaker.view(batch_size, feature_size)
            else:
                speaker_flat = speaker
            # Now ensure it's [1, features]
            if len(speaker_flat.shape) == 1:
                speaker_normalized = speaker_flat.unsqueeze(0)
            elif speaker_flat.shape[0] > 1:
                speaker_normalized = speaker_flat[0:1].clone()
            else:
                speaker_normalized = speaker_flat.clone()
        
        # Final validation
        if len(speaker_normalized.shape) != 2:
            raise RuntimeError(f"speaker_normalized must be 2D for s2a.generate, got shape {speaker_normalized.shape} (original: {original_speaker_shape})")
        logging.info(f"Normalized speaker embedding once: shape {speaker_normalized.shape} (will be reused for all {len(chunks)} chunks)")
        
        for i, (start_pos, end_pos, chunk_text) in enumerate(chunks):
            logging.info(f"Generating chunk {i+1}/{len(chunks)}: '{chunk_text[:50]}...' ({end_pos - start_pos} chars)")
            
            # Track steps for this chunk
            chunk_steps = {'t2s_steps': 0, 's2a_steps': 0}
            
            def track_chunk_steps():
                # This gets called for both t2s and s2a, but we can't distinguish
                # Just track total steps
                chunk_steps['s2a_steps'] += 1
            
            # Generate semantic tokens (stoks) from text
            # Note: We don't use stoks_prompt because it causes text repetition
            # (t2s.generate prepends stoks_prompt, which duplicates tokens)
            # Instead, we rely on atoks_prompt for audio-level continuity
            stoks_result = pipe.t2s.generate(
                chunk_text, 
                cps=cps, 
                lang=lang, 
                stoks_prompt=None,  # Disabled to prevent text repetition
                step=track_chunk_steps
            )
            stoks = stoks_result[0] if isinstance(stoks_result, tuple) else stoks_result
            
            # Log original shapes for debugging
            logging.info(f"  Raw stoks shape: {stoks.shape}, speaker shape: {speaker_normalized.shape}")
            
            # s2a.generate() expects:
            # - stoks: 1D [sequence] - it uses len(stoks) and adds batch dim internally with .unsqueeze(0)
            # - speakers: 2D [batch, features] - it does speakers.repeat(bs, 1) which requires 2D
            original_stoks_shape = stoks.shape
            
            # Normalize stoks to 1D [sequence] for s2a.generate
            if len(stoks.shape) == 1:
                # Already 1D [sequence] - perfect
                stoks_final = stoks
            elif len(stoks.shape) == 2:
                # 2D [batch, sequence] - remove batch dimension, take first batch
                if stoks.shape[0] > 1:
                    stoks_final = stoks[0]  # Take first batch, removes batch dim -> [sequence]
                else:
                    stoks_final = stoks.squeeze(0)  # Remove batch dim -> [sequence]
            else:
                # 3D+ - flatten to 1D by taking first batch and flattening
                logging.warning(f"  Unexpected stoks shape {stoks.shape}, flattening to 1D")
                # Take first batch and ensure 1D
                while len(stoks.shape) > 1:
                    stoks = stoks[0] if stoks.shape[0] == 1 else stoks[0]
                stoks_final = stoks
            
            # Use pre-normalized speaker embedding (reused for all chunks)
            speaker_final = speaker_normalized
            
            # Final validation
            if len(stoks_final.shape) != 1:
                raise RuntimeError(f"stoks_final must be 1D for s2a.generate, got shape {stoks_final.shape} (original: {original_stoks_shape})")
            if len(speaker_final.shape) != 2:
                raise RuntimeError(f"speaker_final must be 2D for s2a.generate, got shape {speaker_final.shape}")
            logging.info(f"  Normalized stoks shape: {stoks_final.shape} (1D for s2a), using pre-normalized speaker: {speaker_final.shape} (2D for s2a)")
            
            # Generate audio tokens (atoks) from semantic tokens
            # Use atoks_prompt from previous chunk for continuity
            atoks = pipe.s2a.generate(
                stoks_final,
                speaker_final,
                langs=None,
                atoks_prompt=atoks_prompt,
                step=track_chunk_steps
            )
            
            # Decode audio tokens to waveform
            chunk_audio = pipe.vocoder.decode(atoks)
            chunk_audio = normalize_audio_shape(chunk_audio)
            
            chunk_duration = chunk_audio.shape[2] / sample_rate if len(chunk_audio.shape) >= 3 else chunk_audio.shape[1] / sample_rate
            logging.info(f"  Chunk {i+1} generated: {chunk_duration:.2f}s ({chunk_steps['s2a_steps']} steps)")
            
            # Trim overlap from the beginning of chunks (except the first)
            # atoks_prompt causes the model to include those tokens in the output, creating overlap
            # We need to trim the overlap to prevent repetition between chunks
            if i > 0 and atoks_prompt is not None:  # Not the first chunk and we used a prompt
                # Estimate overlap duration based on prompt_tokens
                # Calculate token count in this chunk's atoks
                if len(atoks.shape) >= 2:
                    total_tokens = atoks.shape[-1]  # Last dimension is tokens
                    if total_tokens > 0:
                        # Estimate overlap as proportion of prompt_tokens to total tokens
                        # Use a conservative estimate: prompt_tokens represents some duration
                        # We'll estimate based on the chunk's token-to-duration ratio
                        tokens_per_second = total_tokens / chunk_duration if chunk_duration > 0 else 0
                        if tokens_per_second > 0:
                            # Estimate overlap duration from prompt_tokens
                            overlap_duration = prompt_tokens / tokens_per_second
                            # Add a buffer (20%) to ensure we trim enough - the model may include
                            # more overlap than just the prompt tokens
                            overlap_duration *= 1.2
                            
                            # Convert to samples
                            overlap_samples = int(overlap_duration * sample_rate)
                            
                            # Trim from the beginning (dim=2 is samples dimension for [batch, channels, samples])
                            if len(chunk_audio.shape) >= 3:
                                total_samples = chunk_audio.shape[2]
                            else:
                                total_samples = chunk_audio.shape[1]
                            
                            if overlap_samples > 0 and overlap_samples < total_samples:
                                # Trim overlap from start
                                if len(chunk_audio.shape) >= 3:
                                    chunk_audio = chunk_audio[:, :, overlap_samples:]
                                else:
                                    chunk_audio = chunk_audio[:, overlap_samples:]
                                
                                new_duration = (total_samples - overlap_samples) / sample_rate
                                logging.info(f"  Trimmed {overlap_duration:.2f}s ({overlap_samples} samples) overlap from chunk {i+1}, new duration: {new_duration:.2f}s")
                            elif overlap_samples >= total_samples:
                                logging.warning(f"  Overlap estimate ({overlap_duration:.2f}s) exceeds chunk duration ({chunk_duration:.2f}s), skipping trim")
            
            audio_chunks.append(chunk_audio)
            
            # Extract last N tokens for next chunk's prompt (sliding window)
            # Only use atoks_prompt for audio-level continuity (stoks_prompt causes text repetition)
            if i < len(chunks) - 1:  # Not the last chunk
                # Extract last prompt_tokens tokens from atoks for audio continuity
                # s2a.generate expects atoks_prompt as [batch, quantizers, tokens]
                # Usage: toks[:,i,1+i:start+i+1] = atoks_prompt[:,i] requires atoks_prompt[:,i] to be [batch, tokens]
                if len(atoks.shape) == 3:
                    # [batch, quantizers, tokens] - extract and keep batch dimension
                    token_dim = atoks.shape[2]
                    if token_dim > prompt_tokens:
                        atoks_prompt = atoks[:, :, -prompt_tokens:]  # Keep batch dim: [batch, quantizers, prompt_tokens]
                    else:
                        atoks_prompt = atoks  # Keep all: [batch, quantizers, tokens]
                elif len(atoks.shape) == 2:
                    # [quantizers, tokens] - add batch dimension
                    token_dim = atoks.shape[1]
                    if token_dim > prompt_tokens:
                        atoks_prompt = atoks[:, -prompt_tokens:].unsqueeze(0)  # Add batch: [1, quantizers, prompt_tokens]
                    else:
                        atoks_prompt = atoks.unsqueeze(0)  # Add batch: [1, quantizers, tokens]
                else:
                    atoks_prompt = None
                
                # Don't use stoks_prompt - it causes text repetition when prepended by t2s.generate
                stoks_prompt = None
                
                logging.info(f"  Extracted atoks_prompt shape: {atoks_prompt.shape if atoks_prompt is not None and hasattr(atoks_prompt, 'shape') else 'N/A'}")
        
        # Concatenate audio chunks along time dimension (dim=2 for [batch, channels, samples])
        if audio_chunks:
            # All chunks should be [1, channels, samples], concatenate along samples dimension
            final_audio = torch.cat(audio_chunks, dim=2)
        else:
            raise RuntimeError("No audio chunks were generated")
        
        # Calculate total duration (dim=2 is samples dimension)
        total_samples = final_audio.shape[2] if len(final_audio.shape) >= 3 else final_audio.shape[1]
        logging.info(f"Generated {len(chunks)} chunks, total audio length: {total_samples / sample_rate:.2f}s")
        
        # Check if audio ends cleanly (near zero crossing) - helps diagnose if model finished properly
        # If audio doesn't end near zero, the model may not have completed generation
        last_samples = final_audio[0, 0, -100:].abs().mean().item()
        if last_samples > 0.01:  # Arbitrary threshold - if last 100 samples average > 0.01
            logging.warning(f"Audio may not have completed cleanly - last 100 samples average amplitude: {last_samples:.4f} (expected near 0)")
        
        return io.NodeOutput({
            "waveform": final_audio,
            "sample_rate": sample_rate
        })


class WhisperSpeech(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WhisperSpeechGenerate,
        ]


async def comfy_entrypoint() -> WhisperSpeech:
    return WhisperSpeech()

