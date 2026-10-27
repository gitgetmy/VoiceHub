"""VoiceHub: local, sequential MLX speech workstation for Apple Silicon."""
from __future__ import annotations
import os, subprocess, threading, uuid
from pathlib import Path
import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
LOCK = threading.Lock()
MLX_PY = ROOT / ".venv/bin/python"
BREEZE = ROOT / "models/Breeze-TTS-2-BF16"
INDEX_ROOT = HERE / "engines/mlx-indextts2"
INDEX_CLI = INDEX_ROOT / ".venv/bin/mlx-indextts"
INDEX_MODEL = ROOT / "models/IndexTTS-2.5-fp16"

def invoke(args, cwd=None, timeout=900):
    p = subprocess.run([str(x) for x in args], cwd=cwd, timeout=timeout,
        text=True, capture_output=True, env={**os.environ, "PYTHONUNBUFFERED":"1"})
    log = (p.stdout + "\n" + p.stderr).strip()
    if p.returncode:
        raise RuntimeError(log[-3500:] or f"Process exited {p.returncode}")
    return log[-1500:]

def synthesize(engine, text, instruction, cfg, breeze_audio, index_audio, transcript, emotion, alpha):
    if not (text or "").strip(): return None, "请输入配音文本。"
    if not LOCK.acquire(blocking=False): return None, "有其他推理任务正在执行。"
    try:
        wav = OUT / f"voice_{uuid.uuid4().hex[:12]}.wav"
        if engine == "Breeze TTS 2 · BF16":
            if not (BREEZE/"config.json").is_file(): return None, f"缺少模型配置：{BREEZE}"
            args = [MLX_PY,"-m","mlx_audio.tts.generate","--model",BREEZE,
                    "--text",text,"--output_path",OUT,"--file_prefix",wav.stem,
                    "--cfg_scale",str(cfg)]
            if instruction.strip(): args += ["--instruct",instruction]
            if breeze_audio:
                if not (transcript or "").strip(): return None, "声音克隆需要参考音频的准确逐字稿。"
                args += ["--ref_audio",Path(breeze_audio).resolve(),"--ref_text",transcript,"--voice","S0"]
            log = invoke(args,cwd=ROOT)
            matches = sorted(OUT.glob(wav.stem+"*.wav"),key=lambda f:f.stat().st_mtime,reverse=True)
            if not matches: return None,"生成结束，但没有找到 WAV。\n"+log
            return str(matches[0]),"完成。\n"+log[-500:]
        if engine == "IndexTTS 2.5 · FP16":
            if not INDEX_CLI.exists(): return None,"IndexTTS 推理环境未安装，请查看 README。"
            if not INDEX_MODEL.exists(): return None,f"IndexTTS 模型未找到：{INDEX_MODEL}"
            if not index_audio: return None,"请上传一段有使用权的参考声音。"
            args = [INDEX_CLI,"generate","--profile","v25","--model",INDEX_MODEL,
                    "--language","zh","--ref-audio",Path(index_audio).resolve(),
                    "--text",text,"--output",wav,"--emotion",emotion,
                    "--emo-alpha",str(alpha),"--no-denoise-ref"]
            log = invoke(args,cwd=INDEX_ROOT)
            return (str(wav) if wav.is_file() else None),("完成。\n" if wav.is_file() else "未生成 WAV。\n")+log[-500:]
        return None,"未识别的模型。"
    except subprocess.TimeoutExpired: return None,"任务超过 15 分钟。"
    except Exception as exc: return None,f"生成失败：{exc}"
    finally: LOCK.release()

def transcribe(engine, audio):
    if not audio: return "请上传音频。",""
    if not LOCK.acquire(blocking=False): return "另一个任务正在执行。",""
    try:
        adapter = "asr_qwen.py" if engine.startswith("Qwen") else "asr_confucius.py"
        log = invoke([MLX_PY,HERE/adapter,Path(audio).resolve()],cwd=HERE)
        marker="===TRANSCRIPT==="
        if marker not in log: return "识别结果格式异常。",log
        detail,content = log.rsplit(marker,1)
        return content.strip(),detail[-700:]
    except subprocess.TimeoutExpired: return "识别超时。",""
    except Exception as exc: return f"识别失败：{exc}",""
    finally: LOCK.release()

SAMPLES={
    "日常问候":("大家好，欢迎体验本地人工智能语音工作台。希望今天的你有个好心情。","自然亲切的中文成年声音，发音清晰，语速适中，轻松交谈。"),
    "知识讲解":("地球是距离太阳第三近的行星，表面拥有丰富的液态水。","专业但自然的中文科普讲解，断句清楚，不要夸张。"),
    "新闻播报":("今天我们关注科技领域的新进展。人工智能正在帮助人们提升工作效率。","成熟稳重的中文播报声线，节奏平衡，语气客观。"),
    "情感旁白":("有些平凡的日子，很多年后再回想，才发现它们藏着珍贵的温暖。","温柔细腻的成年中文旁白，稍慢，略有感慨但不过分煽情。"),
    "客服提示":("您好，您的申请已经收到。我们会尽快处理，并通过消息通知您。","礼貌温和的服务声音，咬字准确，避免机械感。"),
}
def example(k): return SAMPLES[k]
def mode(value):
    breeze=value.startswith("Breeze")
    return gr.update(visible=breeze),gr.update(visible=not breeze),gr.update(value=None),gr.update(value="")
def dispatch(engine,text,guide,cfg,b_ref,i_ref,reftext,emotion,alpha):
    return synthesize(engine,text,guide,cfg,b_ref,i_ref,reftext,emotion,alpha)

CSS="""
.gradio-container{max-width:1250px!important;margin:auto!important;padding:18px!important}
#hero{padding:30px;border-radius:22px;background:linear-gradient(110deg,#16243c,#1b3e48);margin-bottom:18px}
#hero *{color:white!important}#hero h1{font-size:2.2rem!important;margin:0!important}
#hero p{color:#d4e7f0!important}
#panel{border:1px solid var(--border-color-primary);border-radius:18px;padding:16px}
.gradio-container button.primary{border-radius:12px!important;font-weight:700!important}
@media(max-width:700px){#hero{padding:20px}#hero h1{font-size:1.5rem!important}}
"""
with gr.Blocks(title="VoiceHub · Local Voice Studio",theme=gr.themes.Soft(primary_hue="teal",neutral_hue="slate"),css=CSS) as demo:
    gr.Markdown("""<div id="hero"><h1>VoiceHub / Local Voice Studio</h1><p>文字转语音 · 声音克隆 · 情绪表达 · 离线语音识别<br>四模型统一入口，单任务运行，模型推理结束后退出进程。</p></div>""")
    with gr.Tabs():
        with gr.Tab("语音合成 · TTS"):
            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Group(elem_id="panel"):
                        gr.Markdown("### 01 / 台词与模型")
                        model=gr.Dropdown(["Breeze TTS 2 · BF16","IndexTTS 2.5 · FP16"],value="Breeze TTS 2 · BF16",label="选择引擎")
                        sample=gr.Dropdown(list(SAMPLES),value="日常问候",label="通用示例")
                        text=gr.Textbox(label="配音台词",lines=6,value=SAMPLES["日常问候"][0])
                    with gr.Group(visible=True,elem_id="panel") as bgroup:
                        gr.Markdown("### 02 / Breeze 声音设计")
                        guide=gr.Textbox(label="声音描述 / 表演指导",lines=3,value=SAMPLES["日常问候"][1])
                        cfg=gr.Slider(1,8,value=4,step=.5,label="CFG 引导强度")
                        b_audio=gr.Audio(type="filepath",sources=["upload","microphone"],label="声音克隆参考音频（可选）")
                        reftext=gr.Textbox(label="参考音频逐字稿（克隆时必填）")
                    with gr.Group(visible=False,elem_id="panel") as igroup:
                        gr.Markdown("### 02 / IndexTTS 角色克隆")
                        i_audio=gr.Audio(type="filepath",sources=["upload","microphone"],label="参考音频（必填）")
                        emotion=gr.Dropdown(["calm","happy","sad","angry","afraid","melancholic","surprised","disgusted"],value="calm",label="情绪")
                        alpha=gr.Slider(0,1,value=.6,step=.05,label="情绪强度")
                        gr.Markdown("仅上传你有权使用的声音。默认跳过 Demucs 降噪。")
                with gr.Column(scale=5):
                    with gr.Group(elem_id="panel"):
                        gr.Markdown("### 03 / 生成与试听")
                        go=gr.Button("生成语音",variant="primary")
                        player=gr.Audio(type="filepath",label="试听与下载 WAV")
                        status=gr.Textbox(label="状态 / 日志",lines=6,show_copy_button=True)
            sample.change(example,sample,[text,guide],queue=False)
            model.change(mode,model,[bgroup,igroup,player,status],queue=False)
            go.click(dispatch,[model,text,guide,cfg,b_audio,i_audio,reftext,emotion,alpha],[player,status],concurrency_limit=1)
        with gr.Tab("语音识别 · ASR"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_id="panel"):
                        gr.Markdown("### 01 / 选择识别模型")
                        asr_model=gr.Dropdown(["Qwen3-ASR 1.7B","Confucius4-R2T2"],value="Qwen3-ASR 1.7B",label="引擎")
                        audio=gr.Audio(type="filepath",sources=["upload","microphone"],label="录制或上传音频")
                        recognize=gr.Button("开始识别",variant="primary")
                with gr.Column(scale=6):
                    with gr.Group(elem_id="panel"):
                        gr.Markdown("### 02 / 识别文本")
                        transcript=gr.Textbox(lines=10,label="转写结果",show_copy_button=True)
                        log=gr.Textbox(lines=4,label="日志")
            recognize.click(transcribe,[asr_model,audio],[transcript,log],concurrency_limit=1)
            gr.Markdown("目前仅支持整段离线转写，Confucius 的实时流式协议暂未集成。")
        with gr.Tab("环境与说明"):
            gr.Markdown("""### 本地模型目录
- \`models/Breeze-TTS-2-BF16\`
- \`models/IndexTTS-2.5-fp16\`
- \`models/Qwen3-ASR-1.7B-bf16\`
- \`models/Confucius4-R2T2-bf16\`

VoiceHub 不分发模型权重。各模型的使用范围由各自许可证决定，代码许可证不自动授权商业使用模型或生成音频。请勿在未获许可情况下克隆他人声音。默认仅监听 localhost。""")
if __name__=="__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1",server_port=int(os.getenv("VOICEHUB_PORT","7860")),share=False)
