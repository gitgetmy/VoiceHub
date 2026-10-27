"""Qwen3-ASR CUDA transcription worker; launched separately for one job."""
import sys
import torch
from qwen_asr import Qwen3ASRModel

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: qwen_worker.py audio_file model_name_or_path")
    _, audio, model_name = sys.argv
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3ASRModel.from_pretrained(
        model_name, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=device, max_new_tokens=256
    )
    result = model.transcribe(audio=audio, language=None)
    print("===TRANSCRIPT===", flush=True)
    print(result[0].text if result else "", flush=True)

if __name__=="__main__":
    main()
