# WhisperSpeech Model Information

## Default Models (when t2s_ref and s2a_ref are empty)

When you leave `t2s_ref` and `s2a_ref` empty (default), WhisperSpeech uses:

### Default T2S Model (Text-to-Semantic)
- **Model**: `t2s-small-en+pl.model`
- **Source**: `collabora/whisperspeech:t2s-small-en+pl.model`
- **Characteristics**: 
  - Balanced quality and speed
  - Supports English and Polish
  - Good general-purpose choice

### Default S2A Model (Semantic-to-Audio)
- **Model**: `s2a-q4-hq-fast-en+pl.model` (likely default)
- **Source**: `collabora/whisperspeech:s2a-q4-hq-fast-en+pl.model`
- **Characteristics**:
  - High quality, fast generation
  - Quantized (q4) for efficiency
  - Supports English and Polish

## Model Compatibility

### ✅ All WhisperSpeech Models Are Compatible

**Important**: All WhisperSpeech T2S and S2A models are designed to work together. You can mix and match any T2S model with any S2A model. They use the same token format and are fully compatible.

### Available Model Options

#### T2S Models (Text-to-Semantic)
1. **`t2s-small-en+pl.model`** (Default)
   - Balanced quality/speed
   - English + Polish

2. **`t2s-fast-small-en+pl.model`**
   - Faster generation
   - English + Polish

3. **`t2s-v1.9-medium-7lang.model`**
   - Higher quality
   - Multilingual (7 languages): **English, French, German, Spanish, Italian, Portuguese, Dutch**
   - ⚠️ **CRITICAL**: Must be paired with a multilingual S2A model (e.g., `s2a-v1.9-medium-7lang.model`)
   - **Using English/Polish S2A with this model causes garbled output** - the S2A can't interpret multilingual semantic tokens
   - ComfyUI will auto-detect and warn if you use a non-multilingual S2A
   - **Recommended pairing**: `t2s-v1.9-medium-7lang.model` + `s2a-v1.9-medium-7lang.model`

4. **`t2s-tiny-en+pl.model`**
   - Fastest, smallest
   - English + Polish

#### S2A Models (Semantic-to-Audio)
1. **`s2a-q4-hq-fast-en+pl.model`** (Recommended Default for English/Polish)
   - High quality, fast
   - Quantized (q4)
   - Supports English + Polish only

2. **`s2a-q4-small-en+pl.model`**
   - Smaller, faster
   - Quantized (q4)

3. **`s2a-v1.1-small-en+pl.model`**
   - Alternative version
   - Smaller model
   - Supports English + Polish only

4. **`s2a-v1.9-medium-7lang.model`** ⚠️ **REQUIRED for multilingual T2S**
   - Multilingual S2A model (7 languages)
   - **MUST be paired with `t2s-v1.9-medium-7lang.model`**
   - Using English/Polish S2A with multilingual T2S causes garbled output

5. **`s2a-v1.9-base-7lang.model`**
   - Multilingual S2A model (base version)
   - Alternative to medium version

6. **`s2a-v1.95-medium-7lang.model`**
   - Multilingual S2A model (newer version)
   - Alternative to v1.9 versions

## Reference Audio / Voice Cloning

### Speaker Embedding Model
- **Model**: SpeechBrain ECAPA-TDNN (not Whisper)
- **Source**: Automatically downloaded from `speechbrain/spkrec-ecapa-voxceleb`
- **Compatibility**: ✅ Works with ALL WhisperSpeech T2S/S2A combinations
- **Purpose**: Extracts speaker characteristics (timbre, pitch, tone) from reference audio

### Important Notes:
1. **Speaker embedding is separate** from T2S/S2A models
2. **All WhisperSpeech models support voice cloning** via speaker embeddings
3. **Accent/pronunciation** comes from the T2S model's training data, NOT the speaker embedding
4. **Voice characteristics** (timbre, pitch) come from the speaker embedding

## Recommended Model Combinations

### High Quality
- **T2S**: `t2s-small-en+pl.model` (Note: `t2s-v1.9-medium-7lang.model` is broken for English)
- **S2A**: `s2a-q4-hq-fast-en+pl.model`

### Fast Generation
- **T2S**: `t2s-tiny-en+pl.model` or `t2s-fast-small-en+pl.model`
- **S2A**: `s2a-q4-small-en+pl.model`

### Balanced (Default)
- **T2S**: `t2s-small-en+pl.model` (default)
- **S2A**: `s2a-q4-hq-fast-en+pl.model` (default)

## Downloading Models

Use the provided script to download models:
```bash
python script_examples/download_whisperspeech_models.py
```

Models are saved to: `models/audio_encoders/`

## Model Location

All WhisperSpeech models are stored in:
- **Path**: `models/audio_encoders/`
- **Format**: `.model` files
- **Naming**: `t2s-*.model` for T2S, `s2a-*.model` for S2A

## Compatibility Summary

✅ **Fully Compatible**:
- Any T2S model + Any S2A model
- All models + Voice cloning (speaker embeddings)
- All models + Reference audio

❌ **NOT Related**:
- WhisperSpeech models are NOT the same as Whisper audio encoder models
- Whisper (for transcription) is separate from WhisperSpeech (for TTS)
- The speaker embedding uses SpeechBrain, not Whisper



