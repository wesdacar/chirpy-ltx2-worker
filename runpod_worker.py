"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: load LTX-2 pipelines and generate video, then upload to R2

Required env vars for R2:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Recommended env vars on RunPod (OPTIONAL if auto-detect works):
- MODEL_PATH=/runpod-volume/models
- TMP_DIR=/runpod-volume/tmp
- TMPDIR=/runpod-volume/tmp
"""

import os
import uuid
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
import runpod

# ------------------------------------------------------------
# Disk auto-detect (prevents /app/tmp and missing /workspace)
# ------------------------------------------------------------

def pick_writable_base() -> str:
    """
    Try common RunPod mount points (serverless/pods differ).
    Returns the first path we can mkdir + write a small temp file into.
    """
    candidates = [
        os.getenv("RUNPOD_VOLUME_PATH", ""),
        "/runpod-volume",
        "/workspace",
        "/volume",
        "/data",
        "/mnt",
        "/tmp",
    ]

    for base in candidates:
        if not base:
            continue
        try:
            Path(base).mkdir(parents=True, exist_ok=True)
            test = Path(base) / ".write_test"
            test.write_text("ok")
            test.unlink()
            return base
        except Exception:
            continue

    return "/tmp"


BASE = pick_writable_base()

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(BASE) / "models"))
TMP_DIR = os.getenv("TMP_DIR", str(Path(BASE) / "tmp"))

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

# Force Python + libs to use big disk temp instead of /app/tmp
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

# -----------------------
# Version + basic config
# -----------------------
WORKER_VERSION = "v-ltx2-full-4"
print(f"✅ Worker booted: {WORKER_VERSION}")
print(f"🗂️ Using MODEL_PATH={MODEL_PATH}")
print(f"🗂️ Using TMP_DIR={TMP_DIR}")

# -----------------------
# R2 upload helper
# -----------------------
def upload_file_to_r2(local_path: str, content_type: str = "video/mp4") -> str:
    required = ["R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing R2 env vars: {', '.join(missing)}")

    endpoint = os.environ["R2_ENDPOINT"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.environ["R2_BUCKET"]

    key = f"ltx2/{uuid.uuid4().hex}.mp4"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    s3.upload_file(
        Filename=local_path,
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=60 * 60,  # 1 hour
    )

# -----------------------
# Smoke test (fast)
# -----------------------
def generate_video_smoke(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt", "Hello")
    out_mp4 = str(Path(TMP_DIR) / f"ltx2_smoke_{uuid.uuid4().hex}.mp4")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=black:s=1280x720:r=24",
            "-t", "1",
            "-pix_fmt", "yuv420p",
            out_mp4
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    video_url = upload_file_to_r2(out_mp4)

    return {
        "success": True,
        "mode": "smoke",
        "message": "R2 upload smoke test succeeded",
        "echo": prompt,
        "video_url": video_url,
        "worker_version": WORKER_VERSION,
    }

# -----------------------
# LTX-2 model files
# -----------------------
LTX2_FILES = [
    (
        "ltx-2-19b-distilled-fp8.safetensors",
        "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors",
    ),
    (
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    ),
    (
        "ltx-2-19b-distilled-lora-384.safetensors",
        "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors",
    ),
]

def ensure_models() -> None:
    """
    Downloads required LTX-2 weights into MODEL_PATH if missing.
    Uses wget -c to resume partial downloads.
    """
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    for fname, url in LTX2_FILES:
        dst = Path(MODEL_PATH) / fname
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and dst.stat().st_size > 0:
            print(f"✅ Already have {fname}")
            continue

        print(f"⬇️ Downloading {fname} -> {dst}")
        subprocess.check_call(["wget", "-c", "-O", str(dst), url])
        print(f"✅ Downloaded {fname}")

# -----------------------
# Import LTX pipelines
# -----------------------
LTX_AVAILABLE = False
LTX_IMPORT_ERROR: Optional[str] = None

TI2VidTwoStagesPipeline = None
DistilledPipeline = None

try:
    # your docker build installs ltx-pipelines => module is ltx_pipelines
    from ltx_pipelines import TI2VidTwoStagesPipeline as _TI2  # type: ignore
    from ltx_pipelines import DistilledPipeline as _DP  # type: ignore

    TI2VidTwoStagesPipeline = _TI2
    DistilledPipeline = _DP
    LTX_AVAILABLE = True
    print("✅ Imported ltx_pipelines")
except Exception as e:
    LTX_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"❌ LTX import failed: {LTX_IMPORT_ERROR}")

# -----------------------
# Pipelines (lazy loaded)
# -----------------------
pipeline_hq = None
pipeline_fast = None

def initialize_models() -> None:
    """
    Initialize LTX-2 pipelines once.

    Different LTX2 builds expose different constructor signatures, so we try a
    few compatible patterns until one works.
    """
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    if pipeline_hq is not None and pipeline_fast is not None:
        return

    print("🧠 Initializing LTX2 pipelines... (constructor API)")
    ensure_models()

    fp8_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-fp8.safetensors")
    upscaler_path = str(Path(MODEL_PATH) / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    lora_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-lora-384.safetensors")

    # ---- HQ pipeline: try multiple signatures
    last_err = None
    attempts = [
        ("kw_spatial_upscaler",
         lambda: TI2VidTwoStagesPipeline(fp8_path, spatial_upscaler=upscaler_path, distilled_lora=lora_path)),
        ("kw_upscaler",
         lambda: TI2VidTwoStagesPipeline(fp8_path, upscaler=upscaler_path, distilled_lora=lora_path)),
        ("positional_3",
         lambda: TI2VidTwoStagesPipeline(fp8_path, upscaler_path, lora_path)),
        ("positional_2_plus_kw_lora",
         lambda: TI2VidTwoStagesPipeline(fp8_path, upscaler_path, distilled_lora=lora_path)),
        ("positional_2",
         lambda: TI2VidTwoStagesPipeline(fp8_path, upscaler_path)),
        ("fp8_only_plus_kw",
         lambda: TI2VidTwoStagesPipeline(fp8_path, distilled_lora=lora_path)),
        ("fp8_only",
         lambda: TI2VidTwoStagesPipeline(fp8_path)),
    ]

    for name, fn in attempts:
        try:
            pipeline_hq = fn()
            print(f"✅ HQ pipeline init OK via: {name}")
            break
        except Exception as e:
            last_err = e
            print(f"⚠️ HQ pipeline init failed via {name}: {type(e).__name__}: {e}")

    if pipeline_hq is None:
        raise RuntimeError(f"Failed to init HQ pipeline. Last error: {type(last_err).__name__}: {last_err}")

    # ---- Fast pipeline: also try multiple signatures
    last_err = None
    fast_attempts = [
        ("kw_lora",
         lambda: DistilledPipeline(fp8_path, distilled_lora=lora_path)),
        ("positional_2",
         lambda: DistilledPipeline(fp8_path, lora_path)),
        ("fp8_only",
         lambda: DistilledPipeline(fp8_path)),
    ]

    for name, fn in fast_attempts:
        try:
            pipeline_fast = fn()
            print(f"✅ Fast pipeline init OK via: {name}")
            break
        except Exception as e:
            last_err = e
            print(f"⚠️ Fast pipeline init failed via {name}: {type(e).__name__}: {e}")

    if pipeline_fast is None:
        raise RuntimeError(f"Failed to init fast pipeline. Last error: {type(last_err).__name__}: {last_err}")

    print(f"✅ Pipelines loaded: hq={type(pipeline_hq)} fast={type(pipeline_fast)}")

def _save_pipeline_output_to_mp4(result: Any) -> str:
    """
    Best-effort saver for whatever the pipeline returns.
    Returns path to MP4 in TMP_DIR.
    """
    out_mp4 = str(Path(TMP_DIR) / f"ltx2_{uuid.uuid4().hex}.mp4")

    if isinstance(result, str) and Path(result).exists():
        Path(result).rename(out_mp4)
        return out_mp4

    if isinstance(result, dict):
        vp = result.get("video_path") or result.get("path") or result.get("mp4")
        if isinstance(vp, str) and Path(vp).exists():
            Path(vp).rename(out_mp4)
            return out_mp4

    if hasattr(result, "save") and callable(getattr(result, "save")):
        result.save(out_mp4)  # type: ignore
        if Path(out_mp4).exists():
            return out_mp4

    raise RuntimeError(f"Unknown pipeline output type: {type(result)} (no path/video_path/save)")

def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    global pipeline_hq, pipeline_fast

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "error": "No prompt provided", "worker_version": WORKER_VERSION}

    duration = int(job_input.get("duration", 5))
    fps = int(job_input.get("fps", 24))
    width = int(job_input.get("width", 1280))
    height = int(job_input.get("height", 720))
    quality = (job_input.get("quality", "high") or "high").lower()

    num_frames = duration * fps
    steps = int(job_input.get("steps", 40 if quality == "high" else 8))
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    enhance_prompt = bool(job_input.get("enhance_prompt", True))

    if pipeline_hq is None or pipeline_fast is None:
        initialize_models()

    selected = pipeline_hq if quality == "high" else pipeline_fast
    if selected is None:
        raise RuntimeError("Selected pipeline is None after initialize_models()")

    print(f"🎬 LTX2 generating: {duration}s {width}x{height} fps={fps} frames={num_frames} steps={steps} quality={quality}")
    start = time.time()

    result = selected(  # type: ignore
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        enhance_prompt=enhance_prompt,
    )

    gen_time = time.time() - start
    print(f"⚡ LTX2 generation finished in {gen_time:.2f}s")

    out_mp4 = _save_pipeline_output_to_mp4(result)
    video_url = upload_file_to_r2(out_mp4)

    return {
        "success": True,
        "mode": "ltx2",
        "video_url": video_url,
        "generation_time": gen_time,
        "worker_version": WORKER_VERSION,
        "meta": {
            "prompt": prompt,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": num_frames,
            "steps": steps,
            "quality": quality,
            "guidance_scale": guidance_scale,
            "enhance_prompt": enhance_prompt,
        },
    }

# -----------------------
# RunPod handler
# -----------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("id", "unknown")
    print(f"📦 Processing job {job_id}")

    try:
        job_input = job.get("input") or {}
        # UI sometimes double-wraps
        if isinstance(job_input, dict) and "input" in job_input and isinstance(job_input["input"], dict):
            job_input = job_input["input"]

        mode = (job_input.get("mode", "smoke") or "smoke").lower()
        print(f"🧭 mode={mode}")

        if mode == "ltx2":
            return generate_video_ltx2(job_input)
        else:
            return generate_video_smoke(job_input)

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "error": str(e),
            "worker_version": WORKER_VERSION,
        }

# IMPORTANT:
# Only start serverless when run as main.
# This prevents Dockerfile "import runpod_worker" checks from starting worker.
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
