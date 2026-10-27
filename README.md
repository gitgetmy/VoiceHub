# VoiceHub · Local AI Voice Studio

A local Gradio dashboard for four speech models on Apple Silicon. Chinese-first UI, generic demo scripts, voice cloning, emotion settings, offline transcription, and one active inference subprocess at a time.

> **Status**: public release candidate. Existing Breeze and IndexTTS CLI inference paths have been user-tested on a Mac mini, and Confucius4 has been user-tested in VoiceHub. This new UI and all model combinations have **not** been end-to-end tested on another machine. Qwen3-ASR requires its downloaded `config.json` and weights.

## Supported engines

| Engine | Function | Precision | Notes |
|---|---|---|---|
| Breeze TTS 2 | TTS, voice design / cloning | MLX BF16 | Requires `mlx-audio`; Breeze model license may restrict commercial use |
| IndexTTS 2.5 | TTS, cloning / emotion | MLX FP16 | Requires separate `mlx-indextts2` CLI and clean authorized reference recording |
| Qwen3-ASR 1.7B | Offline speech recognition | MLX BF16 | Using `mlx_audio.stt.utils.load_model` |
| Confucius4-R2T2 | Offline speech recognition | MLX BF16 | **Not** implementing real-time streaming |

## 1. Directory layout

The repository content is intended to sit inside a parent folder named anything you like (examples call it `BreezeTTS`). The launch command is run from that parent folder.

```text
BreezeTTS/
├── .venv/                             # your MLX-Audio venv
├── models/                            # model files; never commit
│   ├── Breeze-TTS-2-BF16/
│   ├── IndexTTS-2.5-fp16/
│   ├── Qwen3-ASR-1.7B-bf16/
│   └── Confucius4-R2T2-bf16/
├── VoiceHub/                           # these repository app files
│   ├── app.py
│   ├── asr_qwen.py
│   ├── asr_confucius.py
│   └── engines/mlx-indextts2/          # independently installed checkout
└── requirements-ui.txt
```

If you clone this repository into `BreezeTTS/`, its `VoiceHub/` directory and `requirements-ui.txt` are already in the correct positions.

## 2. Prepare MLX-Audio (Python 3.11, Apple Silicon)

```bash
cd ~/AI/BreezeTTS
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -U mlx-audio
python -m pip install -r requirements-ui.txt
```

If `.venv` is already working, **keep it**. The instructions above are for a new setup. Verify with `python -c 'import mlx_audio, gradio'`. Use supported Python and native ARM64 Python; `python3` may accidentally point at an incompatible version.

## 3. Get the model weights

Use Hugging Face `hf download` (install `huggingface_hub` if needed). These **are not part of this repository**:

```bash
hf download mlx-community/Breeze-TTS-2-mlx --local-dir models/Breeze-TTS-2-BF16
hf download vanch007/mlx-indextts2-2.5-fp16 --local-dir models/IndexTTS-2.5-fp16
hf download mlx-community/Qwen3-ASR-1.7B-bf16 --local-dir models/Qwen3-ASR-1.7B-bf16
hf download mlx-community/Confucius4-R2T2-bf16 --local-dir models/Confucius4-R2T2-bf16
```

Confirm the desired model directory contains `config.json` (and weights) before attempting inference. IndexTTS might fetch additional W2V-BERT/CAMPPlus files to Hugging Face's normal cache on first run.

## 4. Install IndexTTS separately

```bash
brew install uv git
mkdir -p VoiceHub/engines
git clone https://github.com/vanch007/mlx-indextts2.git VoiceHub/engines/mlx-indextts2
cd VoiceHub/engines/mlx-indextts2
uv sync --python /opt/homebrew/bin/python3.11 --extra v25
# Optional denoise extra is deliberately not necessary for this app.
.venv/bin/mlx-indextts --help
cd ../../..
```

IndexTTS is launched with `--no-denoise-ref` by default to avoid Demucs/TorchCodec compatibility issues on some Macs. Use clean audio (and obtain consent for voice cloning).

## 5. Launch

```bash
cd ~/AI/BreezeTTS
source .venv/bin/activate
python VoiceHub/app.py
```

Open <http://127.0.0.1:7860>. Set another port using `VOICEHUB_PORT=7861 python VoiceHub/app.py`. Only localhost access is enabled.

## 6. Test checklist

1. Choose **Breeze BF16**, pick the **日常问候** demo, generate and play WAV.
2. Choose **IndexTTS 2.5**, upload authorized clean speech, select **calm**, generate WAV.
3. Switch to **ASR**, transcribe a short clip with **Confucius4**.
4. Switch to **Qwen3-ASR**, transcribe the same clip; check `models/Qwen3-ASR-1.7B-bf16/config.json` if missing.
5. Confirm only one inference job starts at a time.

## Privacy, security and licenses

- Gradio binds only to `127.0.0.1`; do not set `share=True` or expose it to the public without access controls.
- Model weights, cloned speech, sample recordings, API tokens, local logs and local virtual environments must not be committed. `.gitignore` excludes standard locations and audio files. **Check `git status` before every push.**
- Each model retains its original license. **Source code licensing is separate from model licenses.** Breeze TTS 2 self-hosted use/output may be noncommercial; check the exact version's license before publishing commercial videos.
- Only clone voices you own or have permission to use.

## Troubleshooting

- `uv` selects Python 3.15 → specify `uv sync --python /opt/homebrew/bin/python3.11 --extra v25`.
- `Could not load torchcodec` → this app uses `--no-denoise-ref` for IndexTTS.
- `cannot access local variable cmd` → use updated `app.py`, where the IndexTTS command is created with all options in one list.
- `Config not found` → model path or download is incomplete; check `config.json`.
- No model audio after CLI success → inspect the **运行状态** textbox; ensure correct weights are on disk.

## License

The VoiceHub orchestration/UI source is provided under MIT. The MLX-Audio, IndexTTS, Confucius4, Qwen3-ASR projects and model weights are independent third-party works under their respective terms.