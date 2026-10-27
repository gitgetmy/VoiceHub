"""VoiceHub Windows preview: isolated local engines, Gradio frontend."""
from __future__ import annotations
import io
import os
import sys
import time
import wave
import json
import uuid
import shutil
import socket
import subprocess
import threading
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
import gradio as gr
import requests

ROOT = Path(__file__).resolve().parent.parent
WIN = Path(__file__).resolve().parent
OUTPUT = WIN / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
BREEZE_URL = "http://127.0.0.1:8137"
QWEN_PYTHON = WIN / ".venv-qwen" / "Scripts" / "python.exe"
QWEN_MODEL = os.environ.get("VOICEHUB_QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")

def check_breeze():
    try:
        with urlopen(BREEZE_URL + "/health", timeout=2) as response:
            return json.load(response)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError("Breeze 服务未启动，请先运行本地 breeze-server，监听 127.0.0.1:8137。") from exc

def make_wav(raw_pcm: bytes, path: Path, sample_rate: int):
    if not raw_pcm or len(raw_pcm) % 2:
        raise ValueError("Breeze 返回了空数据或不完整 PCM 数据。")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(raw_pcm)

def breeze_generate(text, mode, instruction, cfg, reference, ref_text):
    if not (text or "").strip():
        return None, "请输入中文或英文合成文本。"
    if not LOCK.acquire(blocking=False):
        return None, "其他推理任务正在执行。"
    try:
        info = check_breeze()
        data = {
            "text": " ".join(text.splitlines()).strip(),
            "cfg_scale": str(cfg),
        }
        if mode != "声音克隆" and instruction.strip():
            data["instruction"] = instruction
        files = None
        handle = None
        if mode != "声音设计":
            if not reference or not ref_text.strip():
                return None, "声音克隆需要参考音频和参考录音原文。"
            handle = open(reference, "rb")
            files = {"ref_audio": (Path(reference).name, handle, "audio/wav")}
            data["ref_text"] = ref_text.strip()
        try:
            response = requests.post(BREEZE_URL + "/v1/audio/speech", data=data, files=files, timeout=900)
            response.raise_for_status()
        finally:
            if handle:
                handle.close()
        sr = int(response.headers.get("X-Sample-Rate", info.get("sample_rate", 24000)))
        output = OUTPUT / f"breeze_{uuid.uuid4().hex[:12]}.wav"
        make_wav(response.content, output, sr)
        return str(output), f"生成成功 · {sr} Hz · {output.name}"
    except Exception as exc:
        return None, "生成失败：" + str(exc)
    finally:
        LOCK.release()

INDEX_CLI = WIN / "engines" / "index-tts" / ".venv" / "Scripts" / "indextts2.exe"
INDEX_MODEL = os.environ.get("VOICEHUB_INDEX_MODEL", str(WIN / "models" / "IndexTTS-2"))

def gpu_status():
    exe = shutil.which("nvidia-smi")
    if not exe: return "未找到 nvidia-smi", None
    try:
        p = subprocess.run([exe, "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        name, free = [v.strip() for v in p.stdout.splitlines()[0].split(",")]
        return f"{name}，空闲 {free} MiB", int(free)
    except Exception as exc:
        return str(exc), None

def index_synthesize(text, reference, emotion, strength):
    if not text.strip() or not reference: return None, "请输入文本并上传授权参考录音。"
    if not INDEX_CLI.exists(): return None, "请先安装 IndexTTS 2 独立 CLI，详见 README。"
    if not LOCK.acquire(blocking=False): return None, "当前有推理任务正在执行。"
    try:
        try:
            check_breeze()
            return None, "Breeze Vulkan 服务还在运行，请停止 breeze-server，释放显存。"
        except RuntimeError:
            pass
        status, free = gpu_status()
        if free is not None and free < 5000:
            return None, f"建议先释放至少 5GB 显存。目前：{status}"
        dest = OUTPUT / f"index_{uuid.uuid4().hex[:12]}.wav"
        cmd = [str(INDEX_CLI), "synth", "--text", text, "--voice", str(Path(reference).resolve()),
               "--output", str(dest), "--model-dir", INDEX_MODEL, "--device", "cuda:0", "--fp16"]
        if emotion != "neutral":
            cmd += ["--emotion-text", emotion, "--emotion-weight", str(strength)]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=900, env={**os.environ, "PYTHONIOENCODING":"utf-8"})
        if p.returncode: return None, "推理失败：\n" + (p.stderr or p.stdout)[-1800:]
        return (str(dest) if dest.is_file() else None), ("生成成功" if dest.is_file() else "未生成 WAV")
    except Exception as exc: return None, f"IndexTTS 出错：{exc}"
    finally: LOCK.release()

def qwen_transcribe(audio):
    if not audio:
        return "", "请上传音频。"
    if not QWEN_PYTHON.is_file():
        return "", "尚未安装 Qwen 独立环境。详见 windows/README.md。"
    if not LOCK.acquire(blocking=False):
        return "", "其他推理任务正在执行。"
    try:
        try:
            check_breeze()
            return "", "请先停止 Breeze 服务后使用 Qwen3-ASR。"
        except RuntimeError:
            pass
        p = subprocess.run(
            [str(QWEN_PYTHON), str(WIN / "qwen_worker.py"), str(Path(audio).resolve()), QWEN_MODEL],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
            env={**os.environ,"PYTHONIOENCODING":"utf-8"},
        )
        if p.returncode:
            return "", "识别失败：\n" + p.stderr[-1800:]
        marker = "===TRANSCRIPT==="
        if marker not in p.stdout:
            return "", "识别程序没有返回可解析文本：\n" + p.stdout[-1200:]
        return p.stdout.split(marker, 1)[1].strip(), "识别完成"
    except Exception as exc:
        return "", str(exc)
    finally:
        LOCK.release()

def system_status():
    try:
        info = check_breeze()
        breeze = f"可用 · {info.get('sample_rate', '未知')} Hz"
    except Exception:
        breeze = "未连接"
    return (
        f"Windows: {sys.platform == 'win32'}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Breeze GGUF 本地服务: {breeze}\n"
        f"Qwen3-ASR 独立环境: {'已安装' if QWEN_PYTHON.is_file() else '未安装'}\n"
        "IndexTTS 2.5 / Confucius4: 待接入独立 Windows 适配器\n"
        "同一时刻只允许一项推理任务"
    )

STYLE = """
.gradio-container { max-width: 1200px!important;margin:auto!important; }
#hero {background:linear-gradient(110deg,#11263a,#164d57); border-radius:20px;
       padding:24px; margin-bottom:18px}
#hero * {color:white!important}
"""
with gr.Blocks(title="VoiceHub Windows Preview") as app:
    gr.Markdown("""<div id="hero"><h1>VoiceHub · Windows Preview</h1>
    <p>本地配音与语音识别 · Windows / CUDA / Vulkan · Beta 开发版</p></div>""")
    with gr.Tabs():
        with gr.Tab("Breeze TTS 2 · 本地合成"):
            with gr.Row():
                with gr.Column():
                    speech = gr.Textbox(label="合成文本 · 中文 / English", lines=5,
                        value="大家好，欢迎体验 VoiceHub。这里可以使用本地模型生成自然的语音。")
                    mode = gr.Radio(["声音设计", "声音克隆", "声音克隆 + 情绪指导"],
                        value="声音设计", label="模式")
                    instruct = gr.Textbox(label="音色 / 情绪指导",
                        value="自然、亲切、成熟的中文声音，语速平稳。")
                    cfg = gr.Slider(0.5, 5, step=0.5, value=1, label="CFG")
                    ref = gr.Audio(type="filepath", sources=["upload"], label="参考音频（克隆时需要）")
                    reftext = gr.Textbox(label="参考录音原文（克隆时需要）")
                with gr.Column():
                    btn = gr.Button("生成语音", variant="primary")
                    result = gr.Audio(type="filepath",label="音频试听 / 下载")
                    log = gr.Textbox(label="状态",lines=4)
            btn.click(breeze_generate,[speech,mode,instruct,cfg,ref,reftext],[result,log],concurrency_limit=1)
        with gr.Tab("Qwen3-ASR 1.7B · 转写"):
            audio = gr.Audio(type="filepath",sources=["upload","microphone"],label="音频")
            transcribe = gr.Button("识别文字",variant="primary")
            words = gr.Textbox(label="转写结果",lines=8)
            qlog = gr.Textbox(label="状态")
            transcribe.click(qwen_transcribe,[audio],[words,qlog],concurrency_limit=1)
        with gr.Tab("IndexTTS 2.5 · CUDA 实验版"):
            gr.Markdown("先停止 Breeze 服务再使用 IndexTTS；此适配器尚待 Windows 实机测试。")
            idx_text=gr.Textbox(label="合成文本",lines=4,value="您好，欢迎使用 VoiceHub。")
            idx_reference=gr.Audio(type="filepath",sources=["upload"],label="参考声音")
            idx_emotion=gr.Dropdown(["neutral","happy","sad","angry","afraid","calm"],
                                    value="neutral",label="情绪")
            idx_strength=gr.Slider(0,1,value=.6,step=.05,label="情绪强度")
            idx_go=gr.Button("生成 IndexTTS 语音",variant="primary")
            idx_output=gr.Audio(type="filepath",label="试听 WAV")
            idx_log=gr.Textbox(label="日志")
            idx_go.click(index_synthesize,[idx_text,idx_reference,idx_emotion,idx_strength],
                         [idx_output,idx_log],concurrency_limit=1)
        with gr.Tab("Confucius4 · 开发中"):
            gr.Markdown("Windows 流式识别仍需独立适配和 GPU 实测。")
        with gr.Tab("环境诊断"):
            diag = gr.Textbox(value=system_status(), label="运行状态", lines=8)
            refresh = gr.Button("刷新状态")
            refresh.click(system_status, outputs=diag)
            gr.Markdown("Breeze API 默认本地端口 **8137**；VoiceHub 页面默认端口 **7860**。不要把没有认证的推理服务暴露到公网。")
if __name__ == "__main__":
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, share=False,
        css=STYLE, theme=gr.themes.Soft(primary_hue="teal"))
