"""Offline transcription using local Confucius4 MLX-Audio weights."""
import sys
from pathlib import Path
from mlx_audio.stt.utils import load_model
model_dir = Path(__file__).resolve().parent.parent / "models/Confucius4-R2T2-bf16"
if not (model_dir / "config.json").exists():
    raise FileNotFoundError(f"Missing config.json under {model_dir}")
model = load_model(str(model_dir))
result = model.generate(sys.argv[1], language="Chinese")
print("===TRANSCRIPT===", flush=True)
print(result.text, flush=True)
