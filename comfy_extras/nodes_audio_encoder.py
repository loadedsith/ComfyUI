import folder_paths
import comfy.audio_encoders.audio_encoders
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class AudioEncoderLoader(io.ComfyNode):
    """
    Loads an audio encoder model from the models/audio_encoders directory.
    
    Supports:
    - wav2vec2 models (base and large variants)
    - Whisper models (tiny, base, small, medium, large-v2, large-v3)
    
    Models should be PyTorch state dicts (.pt, .safetensors, etc.).
    Model type is automatically detected from the state dict structure.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioEncoderLoader",
            category="loaders",
            inputs=[
                io.Combo.Input(
                    "audio_encoder_name",
                    options=folder_paths.get_filename_list("audio_encoders"),
                    tooltip="Select an audio encoder model file from models/audio_encoders/"
                ),
            ],
            outputs=[io.AudioEncoder.Output()],
        )

    @classmethod
    def execute(cls, audio_encoder_name) -> io.NodeOutput:
        audio_encoder_path = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder_name)
        sd = comfy.utils.load_torch_file(audio_encoder_path, safe_load=True)
        try:
            audio_encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)
        except RuntimeError as e:
            # Add helpful hints to error message
            hint = comfy.audio_encoders.audio_encoders.audio_encoder_detection_error_hint(audio_encoder_path, sd)
            raise RuntimeError(str(e) + hint) from e
        
        if audio_encoder is None:
            hint = comfy.audio_encoders.audio_encoders.audio_encoder_detection_error_hint(audio_encoder_path, sd)
            raise RuntimeError(f"ERROR: audio encoder file '{audio_encoder_name}' is invalid and does not contain a valid model.{hint}")
        return io.NodeOutput(audio_encoder)


class AudioEncoderEncode(io.ComfyNode):
    """
    Encodes audio using the loaded audio encoder model.
    
    Automatically handles long audio by chunking when audio exceeds the model's
    maximum context window. Chunks are processed independently and concatenated
    without overlap to avoid duplication.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioEncoderEncode",
            category="conditioning",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
                io.Float.Input(
                    "max_chunk_length",
                    default=None,
                    min=1.0,
                    max=300.0,
                    optional=True,
                    tooltip="Maximum chunk length in seconds for processing long audio. "
                           "If None, uses the model's default context window (~30s). "
                           "Audio longer than this will be split into non-overlapping chunks."
                ),
            ],
            outputs=[io.AudioEncoderOutput.Output()],
        )

    @classmethod
    def execute(cls, audio_encoder, audio, max_chunk_length=None) -> io.NodeOutput:
        output = audio_encoder.encode_audio(audio["waveform"], audio["sample_rate"], max_chunk_length=max_chunk_length)
        return io.NodeOutput(output)


class AudioEncoder(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AudioEncoderLoader,
            AudioEncoderEncode,
        ]


async def comfy_entrypoint() -> AudioEncoder:
    return AudioEncoder()
