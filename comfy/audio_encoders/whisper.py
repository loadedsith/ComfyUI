import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional
from comfy.ldm.modules.attention import optimized_attention_masked
import comfy.ops

class WhisperFeatureExtractor(nn.Module):
    def __init__(self, n_mels=128, device=None, max_chunk_length=None):
        super().__init__()
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = n_mels
        # max_chunk_length in seconds, None means no limit (for chunking at higher level)
        self.max_chunk_length = max_chunk_length
        self.n_samples = int(max_chunk_length * self.sample_rate) if max_chunk_length else None

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=0,
            f_max=8000,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

    def __call__(self, audio):
        audio = torch.mean(audio, dim=1)
        batch_size = audio.shape[0]
        processed_audio = []

        for i in range(batch_size):
            aud = audio[i]
            if self.n_samples is not None:
                # Only pad/truncate if max_chunk_length is set
                if aud.shape[0] > self.n_samples:
                    aud = aud[:self.n_samples]
                elif aud.shape[0] < self.n_samples:
                    aud = F.pad(aud, (0, self.n_samples - aud.shape[0]))
            # If n_samples is None, process audio as-is (for chunking at higher level)
            processed_audio.append(aud)

        audio = torch.stack(processed_audio)

        mel_spec = self.mel_spectrogram(audio.to(self.mel_spectrogram.spectrogram.window.device))[:, :, :-1].to(audio.device)

        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0

        return log_mel_spec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dtype=None, device=None, operations=None):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = operations.Linear(d_model, d_model, dtype=dtype, device=device)
        self.k_proj = operations.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(d_model, d_model, dtype=dtype, device=device)
        self.out_proj = operations.Linear(d_model, d_model, dtype=dtype, device=device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_output = optimized_attention_masked(q, k, v, self.n_heads, mask)
        attn_output = self.out_proj(attn_output)

        return attn_output


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dtype=None, device=None, operations=None):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dtype=dtype, device=device, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(d_model, dtype=dtype, device=device)

        self.fc1 = operations.Linear(d_model, d_ff, dtype=dtype, device=device)
        self.fc2 = operations.Linear(d_ff, d_model, dtype=dtype, device=device)
        self.final_layer_norm = operations.LayerNorm(d_model, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, attention_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = residual + x

        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        n_ctx: int = 1500,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()

        self.conv1 = operations.Conv1d(n_mels, n_state, kernel_size=3, padding=1, dtype=dtype, device=device)
        self.conv2 = operations.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1, dtype=dtype, device=device)

        self.embed_positions = operations.Embedding(n_ctx, n_state, dtype=dtype, device=device)

        self.layers = nn.ModuleList([
            EncoderLayer(n_state, n_head, n_state * 4, dtype=dtype, device=device, operations=operations)
            for _ in range(n_layer)
        ])

        self.layer_norm = operations.LayerNorm(n_state, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = x.transpose(1, 2)

        x = x + comfy.ops.cast_to_input(self.embed_positions.weight[:, :x.shape[1]], x)

        all_x = ()
        for layer in self.layers:
            all_x += (x,)
            x = layer(x)

        x = self.layer_norm(x)
        all_x += (x,)
        return x, all_x


class WhisperModel(nn.Module):
    """
    Generalized Whisper audio encoder model supporting multiple model sizes.
    
    Supports all Whisper model variants:
    - tiny: n_audio_state=384, n_audio_head=6, n_audio_layer=4
    - base: n_audio_state=512, n_audio_head=8, n_audio_layer=6
    - small: n_audio_state=768, n_audio_head=12, n_audio_layer=12
    - medium: n_audio_state=1024, n_audio_head=16, n_audio_layer=24
    - large-v2/v3: n_audio_state=1280, n_audio_head=20, n_audio_layer=32
    """
    def __init__(
        self,
        n_mels: int = 128,
        n_audio_ctx: int = 1500,
        n_audio_state: int = 1280,
        n_audio_head: int = 20,
        n_audio_layer: int = 32,
        dtype=None,
        device=None,
        operations=None,
        max_chunk_length=None
    ):
        super().__init__()
        self.n_audio_ctx = n_audio_ctx
        self.hop_length = 160  # From WhisperFeatureExtractor
        self.sample_rate = 16000

        self.feature_extractor = WhisperFeatureExtractor(n_mels=n_mels, device=device, max_chunk_length=max_chunk_length)

        self.encoder = AudioEncoder(
            n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
            dtype=dtype, device=device, operations=operations
        )

    def get_max_audio_samples(self):
        """
        Calculate maximum audio samples that can be processed in one chunk.
        Based on n_audio_ctx (max mel frames) and hop_length.
        """
        # n_audio_ctx is the max number of mel frames after conv downsampling
        # After conv stride=2, we get n_audio_ctx frames from 2*n_audio_ctx mel frames
        # Each mel frame represents hop_length audio samples
        max_mel_frames = self.n_audio_ctx * 2  # Account for conv stride=2
        max_samples = max_mel_frames * self.hop_length
        return max_samples

    def forward(self, audio):
        mel = self.feature_extractor(audio)
        x, all_x = self.encoder(mel)
        return x, all_x


# Alias for backward compatibility
WhisperLargeV3 = WhisperModel
