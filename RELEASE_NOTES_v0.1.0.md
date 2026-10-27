# VoiceHub v0.1.0 Beta — Release Notes

VoiceHub is a local-first AI voice studio for Apple Silicon macOS, providing a unified Gradio UI for speech synthesis, voice cloning, emotion control and offline speech transcription.

## Included engines
- Breeze TTS 2 BF16: text-to-speech, voice design, voice cloning and direction; Chinese and English text.
- IndexTTS 2.5 FP16: reference voice cloning and emotion controls; W2V-BERT/CAMPPlus caches required for offline inference.
- Qwen3-ASR 1.7B BF16: offline audio transcription.
- Confucius4-R2T2 BF16: offline transcription (real-time streaming is not yet integrated).

## Improvements
- Model-specific controls and generic sample texts.
- Breeze three-mode selector, reference-audio transcription assisted by ASR, and editable reference text.
- Multiline Breeze text normalization, ordered WAV output fallback, and single-task scheduling.
- Local-only server bind and WAV preview/download.

## Requirements
- Apple Silicon Mac, macOS, Python 3.11.
- A working MLX-Audio environment and separate mlx-indextts2 v2.5 environment.
- Model weights downloaded separately and stored outside Git.
- Tested on one Apple Silicon Mac mini; other device and dependency combinations may need adjustments.

## Licensing and privacy
The VoiceHub application code is MIT licensed. Model weights and generated outputs remain subject to their respective upstream licenses; in particular check restrictions applicable to Breeze before commercial use. Obtain consent for voice cloning. No weights, user recordings, or generated audio are included.

## Known limitations
- Qwen3-ASR model must be fully downloaded, including config.json.
- Confucius is batch/offline only in this release.
- Cloning results and speaker consistency depend on reference audio and model behavior.

Full installation instructions: [README](README.md).
