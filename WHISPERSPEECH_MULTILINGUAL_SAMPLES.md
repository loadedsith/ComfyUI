# WhisperSpeech Multilingual Model Notes

The multilingual WhisperSpeech checkpoints work, but they must be paired with *matching* semantic-to-audio (`s2a`) and semantic-token (`stoks`) models. Mismatching any of these components results in silence, owl‑hoot noises, or otherwise garbled speech.

## Required files

| Component | Filename | Size | SHA256 |
| --- | --- | --- | --- |
| T2S | `t2s-v1.9-medium-7lang.model` | ~1.52 GB | `8d2f9203d0192049384a796353b58617924541e5199bfb112c1a12fbc9fc59bc` |
| S2A | `s2a-v1.9-medium-7lang.model` *(or `s2a-v1.9-base-7lang.model`)* | ~1.53 GB | `f9ae4853e47930691f74ebc970edd2452a3baeef59cf7c2288c33c09479f83aa` |
| Stoks quantizer | `whisper-vq-stoks-v3-7lang.model` | ~1.32 GB | `5211aeb3ab2fb64c9ebaffd8225cba8e74271660a35040b52d7ab010ba62760b` |

Download them into `models/audio_encoders/` (the same folder ComfyUI lists for WhisperSpeech models). On Linux/macOS you can double-check the bits with:

```bash
ls -lh models/audio_encoders/t2s-v1.9-medium-7lang.model \
       models/audio_encoders/s2a-v1.9-medium-7lang.model \
       models/audio_encoders/whisper-vq-stoks-v3-7lang.model
sha256sum models/audio_encoders/t2s-v1.9-medium-7lang.model \
          models/audio_encoders/s2a-v1.9-medium-7lang.model \
          models/audio_encoders/whisper-vq-stoks-v3-7lang.model
```

Anything significantly smaller (for example 100–200 bytes) is just a huggingface pointer and will yield gibberish.

## Usage notes

1. Set `t2s_ref` to `t2s-v1.9-medium-7lang.model`.
2. **Always** pair it with a multilingual S2A (`s2a-v1.9-medium-7lang.model` or `s2a-v1.9-base-7lang.model`). Mixing in the English/Polish S2A checkpoints is now blocked in the node because it creates non-linguistic output.
3. Place `whisper-vq-stoks-v3-7lang.model` in `models/audio_encoders/` so reference-audio voice cloning uses the 7‑language semantic tokens.
4. Set `lang` to the desired ISO code (`fr`, `de`, `es`, `it`, `pt`, `nl`, etc.).

## Sample prompts (copy/paste into the node to test)

## French (fr)

**Simple:**
```
Bonjour, comment allez-vous aujourd'hui? Je suis heureux de vous rencontrer.
```

**Medium:**
```
Le système de synthèse vocale convertit le texte en parole naturelle. La technologie moderne permet une qualité audio exceptionnelle.
```

## German (de)

**Simple:**
```
Guten Tag, wie geht es Ihnen? Es freut mich, Sie kennenzulernen.
```

**Medium:**
```
Die Sprachsynthese wandelt Text in natürliche Sprache um. Moderne Technologie ermöglicht außergewöhnliche Audioqualität.
```

## Spanish (es)

**Simple:**
```
Hola, ¿cómo estás hoy? Me alegra conocerte.
```

**Medium:**
```
El sistema de síntesis de voz convierte el texto en habla natural. La tecnología moderna permite una calidad de audio excepcional.
```

## Italian (it)

**Simple:**
```
Ciao, come stai oggi? Sono felice di conoscerti.
```

**Medium:**
```
Il sistema di sintesi vocale converte il testo in linguaggio naturale. La tecnologia moderna consente una qualità audio eccezionale.
```

## Portuguese (pt)

**Simple:**
```
Olá, como você está hoje? Prazer em conhecê-lo.
```

**Medium:**
```
O sistema de síntese de voz converte texto em fala natural. A tecnologia moderna permite qualidade de áudio excepcional.
```

## Dutch (nl)

**Simple:**
```
Hallo, hoe gaat het vandaag? Leuk je te ontmoeten.
```

**Medium:**
```
Het spraaksynthesesysteem zet tekst om in natuurlijke spraak. Moderne technologie maakt uitzonderlijke audiokwaliteit mogelijk.
```

## Usage in ComfyUI

1. Set `t2s_ref` to `t2s-v1.9-medium-7lang.model`.
2. Pick `s2a_ref` = `s2a-v1.9-medium-7lang.model` *(or `s2a-v1.9-base-7lang.model`)*—the node now errors if you try to mix multilingual T2S with the English/Polish S2A checkpoints.
3. Drop `whisper-vq-stoks-v3-7lang.model` into `models/audio_encoders/` so reference audio cloning sticks to the same semantic vocabulary.
4. Set `lang` to one of: `fr`, `de`, `es`, `it`, `pt`, or `nl`.
5. Paste one of the sample texts above and generate.

## Expected behavior

- ✅ Produces intelligible speech for the supported languages when all three multilingual files are present.
- ⚠️ Language ID can still drift; explicitly set `lang` to keep it anchored.
- ⚠️ English output sounds different from the EN/PL models and may feel “accented,” which is expected.
