"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: load LTX-2 pipelines and generate video, then upload to R2
"""

import os
import uuid
import time
import subprocess
from pathlib import Path

import boto3
from botocore.config import Config
import runpod

# -----------------------
# Version + basic config
# -----------------------
WORKER_VERSION = "v-ltx2-full-2"
print(f"✅ Worker booted: {WORKER_VERSION}")

# Where we store large model files on RunPod
# You said you set: MODEL_PATH=/workspace/models
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models")
TMP_DIR = os.getenv("TMP_DIR", "/workspace/tmp")

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------
# R2 upload helper
# -----------------------
def upload_file_to_r2(local_path: str, content_type: str = "video/mp4") -> str:
    """
    Uploads a file to Cloudflare R2 (S3-compatible) and returns a presigned URL.
    Requires env vars:
      R2_ENDPOINT
      R2_ACCESS_KEY_ID
      R2_SECRET_ACCESS_KEY
      R2_BUCKET
    """
    missing = [k for k in ["R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"] if not os.getenv(k)]
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

    # Create a tiny 1-second MP4 (black screen) so we can test R2 upload fast.
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
# LTX-2: model files + pipeline loading
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

def ensure_models():
    """
    Downloads required LTX-2 files into MODEL_PATH if missing.
    (These are listed as required in the LTX-2 repo README.)  [oai_citation:1‡GitHub](https://github.com/Lightricks/LTX-2)
    """
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    for fname, url in LTX2_FILES:
        dst = Path(MODEL_PATH) / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"✅ Already have {fname}")
            continue

        print(f"⬇️ Downloading {fname} -> {dst}")
        # -c resumes partial downloads; avoids re-downloading huge files.
        subprocess.check_call(["wget", "-c", "-O", str(dst), url])
        print(f"✅ Downloaded {fname}")

# Import LTX pipelines (underscore module!)
LTX_AVAILABLE = False
LTX_IMPORT_ERROR = None
TI2VidTwoStagesPipeline = None
DistilledPipeline = None

try:
    # Correct package name after installing ltx-pipelines
    from ltx_pipelines import TI2VidTwoStagesPipeline, DistilledPipeline  # type: ignore
    LTX_AVAILABLE = True
    print("✅ Imported ltx_pipelines")
except Exception as e:
    LTX_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"❌ LTX import failed: {LTX_IMPORT_ERROR}")

# Pipelines are loaded lazily (first request in ltx2 mode)
pipeline_hq = None
pipeline_fast = None

def initialize_models():
    """
    Loads HQ + fast pipelines once.
    Uses from_pretrained if available (official examples use it),
    otherwise raises with a clear error so we know what methods exist.
    """
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    # Guard against repeated loads
    if pipeline_hq is not None and pipeline_fast is not None:
        return

    print("🧠 Initializing LTX2 pipelines...")

    # We’re loading from local directory containing the safetensors + assets.
    # Some versions expect a “model id” (HF repo) instead; if this fails,
    # we’ll print the available attributes to decide the correct call.
    if not hasattr(TI2VidTwoStagesPipeline, "from_pretrained"):
        raise RuntimeError(
            "TI2VidTwoStagesPipeline has no from_pretrained(). "
            f"Available attrs: {', '.join(sorted(dir(TI2VidTwoStagesPipeline))[:60])} ..."
        )

    # Try a “local path” load first
    try:
        pipeline_hq = TI2VidTwoStagesPipeline.from_pretrained(
            MODEL_PATH,
            device_map="auto",
        )
    except TypeError:
        # If signature differs, retry minimal
        pipeline_hq = TI2VidTwoStagesPipeline.from_pretrained(MODEL_PATH)

    # Distilled (fast) pipeline
    if not hasattr(DistilledPipeline, "from_pretrained"):
        raise RuntimeError("DistilledPipeline has no from_pretrained() — unexpected install/version mismatch.")

    try:
        pipeline_fast = DistilledPipeline.from_pretrained(
            MODEL_PATH,
            device_map="auto",
        )
    except TypeError:
        pipeline_fast = DistilledPipeline.from_pretrained(MODEL_PATH)

    print(f"✅ Pipelines loaded: hq={type(pipeline_hq)} fast={type(pipeline_fast)}")

def generate_video_ltx2(job_input: dict) -> dict:
    """
    Text-to-video (initial implementation).
    Returns an MP4 uploaded to R2.
    """
    global pipeline_hq, pipeline_fast

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "error": "No prompt provided", "worker_version": WORKER_VERSION}

    # Params
    duration = int(job_input.get("duration", 5))
    fps = int(job_input.get("fps", 24))
    width = int(job_input.get("width", 1280))
    height = int(job_input.get("height", 720))
    quality = (job_input.get("quality", "high") or "high").lower()

    num_frames = duration * fps
    steps = int(job_input.get("steps", 40 if quality == "high" else 8))

    # Ensure model weights exist
    ensure_models()

    # Lazy-load pipelines
    if pipeline_hq is None or pipeline_fast is None:
        initialize_models()

    selected = pipeline_hq if quality == "high" else pipeline_fast
    if selected is None:
        raise RuntimeError("Selected pipeline is None after initialize_models()")

    print(f"🎬 LTX2 generating: {duration}s {width}x{height} fps={fps} frames={num_frames} steps={steps} quality={quality}")
    start = time.time()

    # Call pipeline
    result = selected(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=float(job_input.get("guidance_scale", 7.5)),
        enhance_prompt=bool(job_input.get("enhance_prompt", True)),
    )

    gen_time = time.time() - start
    print(f"⚡ LTX2 generation finished in {gen_time:.2f}s")

    # ---- Saving logic (robust “best effort”)
    out_mp4 = str(Path(TMP_DIR) / f"ltx2_{uuid.uuid4().hex}.mp4")

    # Common cases:
    # - result may be a path string
    # - result may be dict with "video_path"
    # - result may have .save(path)
    if isinstance(result, str) and Path(result).exists():
        Path(result).rename(out_mp4)
    elif isinstance(result, dict) and isinstance(result.get("video_path"), str) and Path(result["video_path"]).exists():
        Path(result["video_path"]).rename(out_mp4)
    elif hasattr(result, "save") and callable(getattr(result, "save")):
        result.save(out_mp4)  # type: ignore
    else:
        # If we get here, we need to inspect what result actually is
        raise RuntimeError(f"Unknown pipeline output type: {type(result)} (no save/video_path)")

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
# This prevents Dockerfile "import runpod_worker" sanity checks from starting serverless.
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
