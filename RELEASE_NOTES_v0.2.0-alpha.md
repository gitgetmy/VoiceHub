# VoiceHub Windows v0.2.0 Alpha

**Experimental source-only release. Windows inference has NOT been validated.**

This release extends the macOS MLX VoiceHub project with a separate Windows Gradio UI and preliminary Vulkan/CUDA adapters. macOS files and the v0.1.0 workflow remain unchanged.

## Status on an RTX 3060 12GB Windows test PC
- Python 3.11 x64 and Windows Gradio web UI started successfully.
- IndexTTS environment installed; PyTorch returned CUDA available = True.
- IndexTTS 2.5 model weights were downloading; **no successful IndexTTS 2.5 speech generation has been demonstrated**.
- Breeze Vulkan and Qwen3-ASR integration code exists, but **no successful inference has been demonstrated**.
- Confucius4 Windows integration is **not implemented**.

## Included source
- `windows/app.py` — local Gradio UI and experimental Breeze / Index / Qwen controls
- `windows/qwen_worker.py` — isolated Qwen Python worker
- `windows/requirements-ui.txt` — UI dependencies
- `windows/README.md` — Windows installation and limitations

## Platform
Windows 11 x64 / Python 3.11 / RTX 3060 12GB test configuration (other GPUs not yet validated).

## Known limitations
- Requires downloading separate upstream models; no model weights or GPU drivers are bundled.
- IndexTTS v2.5 weight compatibility with the current CLI adapter is **unverified**.
- Breeze Vulkan server is separately managed; model unloading is not fully automatic.
- Model paths, API semantics, audio formats, CUDA versions may require adjustments.
- Confucius4 is not yet integrated.
- This alpha is not suitable for unattended production usage.

## License & safety
VoiceHub source code MIT; third-party model licenses apply independently, and Breeze may limit commercial usage. Obtain permission before cloning voices. Bind the local services to 127.0.0.1 and do not expose them publicly without authentication.

[Windows installation guide](windows/README.md)
