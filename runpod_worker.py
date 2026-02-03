"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: download models (once), initialize LTX-2 pipelines, generate MP4, upload to R2

Required RunPod env vars:
- MODEL_PATH=/workspace/models
- TMP_DIR=/workspace/tmp   (optional, defaults below)

R2 env vars:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET
"""

import os
import uuid
import time
import inspect
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.config import Config
import runpod

# -----------------------
# Version + basic config
# -----------------------
WORKER_VERSION = "v-ltx2-full-3"
print(f"✅ Worker booted: {WORKER_VERSION}")

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models")
TMP_DIR = os.getenv("TMP_DIR", "/workspace/tmp")
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

# Optional: keep HF caches on big disk as well
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# -----------------------
# R2 upload helper
# -----------------------
def upload_file_to_r2(local_path: str, content_type: str = "video/mp4") -> str:
    missing = [
        k for k in ["R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
        if not os.getenv(k)
    ]
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
def generate_video_smoke(job_input: dict) -> dict:
    prompt = job_input.get("prompt", "Hello")
    out_mp4 = str(Path(TMP_DIR) / "ltx2_smoke_test.mp4")

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
# LTX-2: model files
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

# Text encoder repo (downloaded by HF automatically into HF cache)
GEMMA_ROOT = os.getenv("GEMMA_ROOT", "google/gemma-3-12b-it-qat-q4_0-unquantized")

def ensure_models() -> Dict[str, str]:
    """
    Ensures all required model files exist locally.
    Returns a dict of paths used by the pipeline constructor.
    """
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    for fname, url in LTX2_FILES:
        dst = Path(MODEL_PATH) / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"✅ Already have {fname}")
        else:
            print(f"⬇️ Downloading {fname} -> {dst}")
            # resume + stable downloads
            subprocess.check_call(["wget", "-c", "-O", str(dst), url])
            print(f"✅ Downloaded {fname}")

        if "distilled-fp8" in fname:
            paths["checkpoint_path"] = str(dst)
        elif "spatial-upscaler" in fname:
            paths["spatial_upsampler_path"] = str(dst)
        elif "distilled-lora" in fname:
            paths["distilled_lora_path"] = str(dst)

    missing_keys = [k for k in ["checkpoint_path", "spatial_upsampler_path", "distilled_lora_path"] if k not in paths]
    if missing_keys:
        raise RuntimeError(f"Missing model paths after download: {missing_keys}")

    return paths

# -----------------------
# LTX pipelines import (underscore module)
# -----------------------
LTX_AVAILABLE = False
LTX_IMPORT_ERROR: Optional[str] = None

TI2VidTwoStagesPipeline = None
DistilledPipeline = None

try:
    from ltx_pipelines import TI2VidTwoStagesPipeline, DistilledPipeline  # type: ignore
    LTX_AVAILABLE = True
    print("✅ Imported ltx_pipelines")
except Exception as e:
    LTX_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"❌ LTX import failed: {LTX_IMPORT_ERROR}")

# -----------------------
# Pipelines (lazy init)
# -----------------------
pipeline_hq = None
pipeline_fast = None

def _safe_construct_pipeline(cls, kwargs: Dict[str, Any]):
    """
    Construct pipeline using only accepted kwargs (filters via signature).
    """
    sig = inspect.signature(cls)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)

def initialize_models() -> None:
    """
    Initializes HQ + fast pipelines using constructor-style API (no from_pretrained).
    """
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    if pipeline_hq is not None and pipeline_fast is not None:
        return

    print("🧠 Initializing LTX2 pipelines... (constructor API)")

    paths = ensure_models()

    # Try a few constructor variants robustly (different package versions vary)
    base_kwargs = {
        "checkpoint_path": paths["checkpoint_path"],
        "distilled_lora_path": paths["distilled_lora_path"],
        "spatial_upsampler_path": paths["spatial_upsampler_path"],
        "gemma_root": GEMMA_ROOT,
        "local_files_only": False,
        "fp8transformer": False,
        "loras": [],
    }

    # HQ pipeline
    try:
        pipeline_hq = _safe_construct_pipeline(TI2VidTwoStagesPipeline, base_kwargs)
    except TypeError as e:
        # Some versions want slightly different names; add fallback attempts
        alt = dict(base_kwargs)
        alt["model_path"] = MODEL_PATH
        pipeline_hq = _safe_construct_pipeline(TI2VidTwoStagesPipeline, alt)

    # Fast pipeline
    try:
        pipeline_fast = _safe_construct_pipeline(DistilledPipeline, base_kwargs)
    except TypeError:
        alt2 = dict(base_kwargs)
        alt2["model_path"] = MODEL_PATH
        pipeline_fast = _safe_construct_pipeline(DistilledPipeline, alt2)

    print(f"✅ Pipelines loaded: hq={type(pipeline_hq)} fast={type(pipeline_fast)}")

def _call_pipeline(pipeline_obj, call_kwargs: Dict[str, Any], out_mp4: str):
    """
    Calls pipeline safely:
    - If pipeline __call__ supports output_path, pass it
    - Else call and handle common return shapes
    """
    call_sig = None
    try:
        call_sig = inspect.signature(pipeline_obj.__call__)
    except Exception:
        call_sig = None

    if call_sig and "output_path" in call_sig.parameters:
        call_kwargs = dict(call_kwargs)
        call_kwargs["output_path"] = out_mp4
        res = pipeline_obj(**call_kwargs)
        # If it wrote to output_path, we’re done.
        if Path(out_mp4).exists() and Path(out_mp4).stat().st_size > 0:
            return out_mp4
        # Otherwise fall through to interpret res.
        return res

    res = pipeline_obj(**call_kwargs)

    # Return formats seen in the wild:
    # - string path
    # - dict with video_path
    # - object with save()
    if isinstance(res, str) and Path(res).exists():
        Path(res).rename(out_mp4)
        return out_mp4

    if isinstance(res, dict):
        vp = res.get("video_path") or res.get("path")
        if isinstance(vp, str) and Path(vp).exists():
            Path(vp).rename(out_mp4)
            return out_mp4

    if hasattr(res, "save") and callable(getattr(res, "save")):
        res.save(out_mp4)  # type: ignore
        return out_mp4

    # If the pipeline returned nothing but wrote a default file, try common defaults
    if Path(out_mp4).exists() and Path(out_mp4).stat().st_size > 0:
        return out_mp4

    raise RuntimeError(f"Unknown pipeline output type: {type(res)} (no output_path/save/video_path)")

def generate_video_ltx2(job_input: dict) -> dict:
    """
    Text-to-video initial implementation.
    Generates MP4 and uploads to R2.
    """
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

    # Ensure pipelines exist
    if pipeline_hq is None or pipeline_fast is None:
        initialize_models()

    selected = pipeline_hq if quality == "high" else pipeline_fast
    if selected is None:
        raise RuntimeError("Selected pipeline is None after initialize_models()")

    print(f"🎬 LTX2 generating: {duration}s {width}x{height} fps={fps} frames={num_frames} steps={steps} quality={quality}")
    start = time.time()

    out_mp4 = str(Path(TMP_DIR) / f"ltx2_{uuid.uuid4().hex}.mp4")

    call_kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        enhance_prompt=enhance_prompt,
    )

    _call_pipeline(selected, call_kwargs, out_mp4)

    gen_time = time.time() - start
    print(f"⚡ LTX2 generation finished in {gen_time:.2f}s")

    if not Path(out_mp4).exists() or Path(out_mp4).stat().st_size == 0:
        raise RuntimeError("MP4 was not created")

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
        },
    }

# -----------------------
# RunPod handler
# -----------------------
def handler(job):
    job_id = job.get("id", "unknown")
    print(f"📦 Processing job {job_id}")

    try:
        job_input = job.get("input") or {}
        # RunPod UI sometimes double-wraps input
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
# Only start the worker when run as the main process.
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
