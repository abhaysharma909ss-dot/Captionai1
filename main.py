import os, uuid, shutil, subprocess
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
app = FastAPI(title="CaptionFlow AI")
model = None

def get_model():
    global model
    if model is None:
        try:
            model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
        except Exception as e:
            raise HTTPException(503, f"Could not load speech model: {e}")
    return model

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "Command failed")[-3000:])
    return p.stdout

def require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise HTTPException(500, "FFmpeg and FFprobe must be installed on the server.")

@app.get("/api/health")
def health():
    return {"status":"ok", "model":MODEL_SIZE, "ffmpeg":bool(shutil.which("ffmpeg"))}

@app.post("/api/transcribe")
async def transcribe(video: UploadFile = File(...), language: Optional[str] = None):
    require_ffmpeg()
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm"}:
        raise HTTPException(400, "Please upload an MP4, MOV or WEBM video.")
    job = uuid.uuid4().hex
    folder = WORK / job
    folder.mkdir()
    src = folder / f"source{suffix}"
    size = 0
    try:
        with src.open("wb") as out:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"Video exceeds {MAX_BYTES//(1024*1024)} MB limit.")
                out.write(chunk)
        audio = folder / "audio.wav"
        run(["ffmpeg","-y","-i",str(src),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(audio)])
        chosen = None if not language or language == "auto" else language
        segments, info = get_model().transcribe(str(audio), language=chosen, vad_filter=True)
        captions = [{"id":i,"start":round(s.start,2),"end":round(s.end,2),"text":s.text.strip()} for i,s in enumerate(segments) if s.text.strip()]
        if not captions:
            raise HTTPException(422, "No speech was detected. Try a video with clearer audio.")
        return {"job_id":job,"language":info.language,"language_probability":round(info.language_probability,3),"duration":info.duration,"captions":captions}
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        await video.close()

class Caption(BaseModel):
    start: float
    end: float
    text: str

class RenderRequest(BaseModel):
    job_id: str
    captions: list[Caption]
    font_family: str = "Arial"
    font_size: int = 42
    text_color: str = "#FFFFFF"
    background: bool = True
    bold: bool = True
    outline: bool = True
    position: str = "bottom"

def ass_time(seconds):
    cs=max(0,int(round(seconds*100)))
    h,rem=divmod(cs,360000); m,rem=divmod(rem,6000); s,c=divmod(rem,100)
    return f"{h}:{m:02}:{s:02}.{c:02}"

def ass_color(value):
    h=value.lstrip("#")
    if len(h)!=6: h="FFFFFF"
    r,g,b=(int(h[i:i+2],16) for i in (0,2,4))
    return f"&H00{b:02X}{g:02X}{r:02X}"

@app.post("/api/render")
def render(req: RenderRequest):
    require_ffmpeg()
    folder=WORK/req.job_id
    sources=list(folder.glob("source.*")) if folder.exists() else []
    if not sources: raise HTTPException(404,"Editing session expired. Upload the video again.")
    if not req.captions: raise HTTPException(400,"Add at least one caption before exporting.")
    ass=folder/"captions.ass"; out=folder/"captionflow-export.mp4"
    font=req.font_family.replace("\\","").replace(",","")[:40]
    size=max(12,min(120,int(req.font_size)))
    color=ass_color(req.text_color); bold=-1 if req.bold else 0
    outline=2 if req.outline else 0; back=3 if req.background else 1
    align={"top":8,"center":5,"bottom":2}.get(req.position,2)
    header=f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{color},&H000000FF,&H00000000,&H99000000,{bold},0,{back},{outline},1,{align},42,42,55,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows=[]
    for c in req.captions:
        start=max(0,float(c.start)); end=max(start+0.1,float(c.end))
        txt=c.text.replace("\\","\\\\").replace("{",r"\{").replace("}",r"\}").replace("\n",r"\N")
        rows.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{txt}")
    ass.write_text(header+"\n".join(rows),encoding="utf-8")
    escaped=str(ass).replace("\\","\\\\").replace(":","\\:").replace("'","\\'")
    try:
        run(["ffmpeg","-y","-i",str(sources[0]),"-vf",f"ass='{escaped}'","-c:v","libx264","-preset","veryfast","-crf","20","-c:a","aac","-b:a","192k","-movflags","+faststart",str(out)])
    except Exception as e:
        raise HTTPException(500,f"Video export failed: {e}")
    return {"download_url":f"/api/download/{req.job_id}"}

@app.get("/api/download/{job_id}")
def download(job_id: str):
    if not job_id.isalnum(): raise HTTPException(400,"Invalid job ID")
    path=WORK/job_id/"captionflow-export.mp4"
    if not path.exists(): raise HTTPException(404,"Export not found.")
    return FileResponse(path,media_type="video/mp4",filename="CaptionFlow-AI.mp4")

app.mount("/",StaticFiles(directory=ROOT/"static",html=True),name="static")
