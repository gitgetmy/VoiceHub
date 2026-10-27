# VoiceHub Windows v0.2.0 Alpha (untested inference preview)

**Release warning:** The Gradio UI has launched successfully on one Windows RTX 3060 12GB computer. **No Windows speech inference has been verified**. IndexTTS 2.5 CUDA dependencies loaded and `torch.cuda.is_available()` returned `True`, but model weights were still downloading. The IndexTTS adapter may require API changes for 2.5 compatibility. This is source-only alpha, not a ready-to-use four-model release.\n\nThis **experimental Windows backend** is isolated from the existing, working Apple Silicon / MLX release. It is **not** yet a full four-engine Windows implementation.

## Current scope

| Engine | Windows implementation | Status |
|---|---|---|
| Breeze TTS 2 | HoppouAI Breeze-TTS-2.cpp Windows Vulkan HTTP server | Adapter written, **Windows testing pending** |
| Qwen3-ASR 1.7B | Qwen official `qwen-asr` PyTorch package (CUDA or CPU) | Adapter written, **Windows testing pending** |
| IndexTTS 2.5 | Official Windows PyTorch CLI (`indextts2 synth`) | Experimental adapter committed; **Windows testing pending** |
| Confucius4-R2T2 | Official streaming/vLLM runtime | **Not integrated yet** |

This preview does not claim working inference for any engine yet. Confucius4 is not integrated. IndexTTS provides an experimental button whose CLI and model version compatibility require validation.

## Prerequisites

- Windows 11, 64-bit
- Python 3.11 (a real Windows installation, not the Apple Silicon MLX environment)
- NVIDIA GPU recommended for Qwen CUDA; Breeze Vulkan can also use AMD/Intel GPUs
- Latest GPU driver
- PowerShell

## 1. Run the dashboard

From the repository root:

```powershell
py -3.11 -m venv windows\.venv
.\windows\.venv\Scripts\python.exe -m pip install -U pip
.\windows\.venv\Scripts\python.exe -m pip install -r windows\requirements-ui.txt
.\windows\.venv\Scripts\python.exe windows\app.py
```

Open http://127.0.0.1:7860 .

## 2. Breeze Vulkan server

Use the **Windows x64 Vulkan** prebuilt release of the independent [Breeze-TTS-2.cpp project](https://github.com/HoppouAI/Breeze-TTS-2.cpp/releases). Download compatible GGUF weights and the `audio_tokenizer/` files described by the upstream project. They are **not** distributed here.

From the extracted server folder:

```powershell
.\breeze-server.exe breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8137
```

You may use another GGUF precision, but match the actual downloaded filename. Verify http://127.0.0.1:8137/health first. Do not bind to `0.0.0.0` or port-forward these unauthenticated services.

The adapter sends a multipart speech request and wraps returned **signed 16-bit mono PCM** as WAV. The `Breeze-TTS-2.cpp` service uses a **different inference engine/weight format** from macOS MLX, so do not copy the MLX weights into the GGUF server.

Voice-cloning multipart behavior is *not yet validated on Windows*; test standard Voice Design before cloning.

## 3. Qwen3-ASR (independent worker environment)

From repository root, PowerShell:

```powershell
py -3.11 -m venv windows\.venv-qwen
.\windows\.venv-qwen\Scripts\python.exe -m pip install -U pip
.\windows\.venv-qwen\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
.\windows\.venv-qwen\Scripts\python.exe -m pip install qwen-asr
```

The CUDA wheel index above is an **example** and must match the user's CUDA-capable GPU/driver and current PyTorch support. Check [PyTorch's official selector](https://pytorch.org/get-started/locally/) before installing. If CPU-only, install the appropriate CPU wheels instead. First use downloads from the official Qwen model name, or set `VOICEHUB_QWEN_MODEL` to the directory containing a compatible **PyTorch** model (not the MLX BF16 conversion).

The worker process exits after transcription, releasing model resources. The UI executes one model job at a time. Qwen initial download requires network access and sufficient disk space.

## 4. IndexTTS 2.5 on RTX 3060 12GB (experimental)

Use the official [IndexTTS2 CLI guide](https://github.com/index-tts/index-tts/blob/main/docs/cli_v2_usage.md). Install the official Windows-compatible project independently under `windows/.venv-index` and make sure `windows/.venv-index/Scripts/indextts2.exe` exists.

A Windows source installation (consult upstream instructions for build dependencies):

```powershell
git clone https://github.com/index-tts/index-tts.git windows/engines/index-tts
cd windows/engines/index-tts
uv sync --python 3.11
cd ../../..
```

After `uv sync`, the preview locates the CLI in `windows/engines/index-tts/.venv/Scripts/indextts2.exe`. Keep its entire virtual environment intact.\n\nDownload required **official PyTorch** IndexTTS2 model resources, not the MLX weights, including `config.yaml`, `gpt.pth`, `s2mel.pth`, and auxiliary files. Set `VOICEHUB_INDEX_MODEL` to that model-resource directory. Upstream CLI includes `indextts2 init`, `download`, and `check`.

Before first VoiceHub launch, validate the CLI itself:
```powershell
indextts2 check --model-dir D:/models/IndexTTS-2 --device cuda:0
indextts2 synth --text "你好，这是 Windows 本地语音。" --voice D:/refs/test.wav --model-dir D:/models/IndexTTS-2 --device cuda:0 --fp16 --output D:/output/test.wav
```

Close the Breeze Vulkan server before invoking IndexTTS or Qwen. VoiceHub checks whether Breeze is still alive, and IndexTTS refuses to start if it can see fewer than 5GB of free video memory. NVIDIA `nvidia-smi` can confirm remaining VRAM. The 5GB threshold is a precaution, **not a guaranteed minimum requirement**.

## Development caveats

- IndexTTS has an experimental subprocess adapter but needs Windows hardware verification; Confucius4 remains disabled.
- Breeze server remains resident while VoiceHub is open; **this preview does not yet implement strict one-model-at-a-time unload across every backend**. Shut it down before heavy Qwen work if needed.
- The Gradio dashboard is locally served, and Qwen runs as a subprocess. No user audio or model weights should be committed to Git.
- Read each upstream model license, including non-commercial restrictions for self-hosted Breeze, before distributing outputs commercially.
