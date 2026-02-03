"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation
"""

import os
import uuid
import time
import subprocess
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
import runpod
import torch

# ---------------------------
# Version / boot marker
# ---------------------------
WORKER_VERSION = "v-clean-1"
print(f"✅ Worker booted: {WORKER_VERSION}")

# ---------------------------
# R2 (S3-compatible) upload
# Required env vars:
#   R2_ENDPOINT
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY
#   R2_BUCKET
# ---------------------------

def upload_file_to_r2(local_path: str, content_type: str = "video/mp4") -> str:
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

# ---------------------------
# Smoke mode (fast sanity test)
# ---------------------------

def generate_video_smoke(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt", "Hello")
    out_mp4 = "/tmp/ltx2_smoke_test.mp4"

    # Create a tiny 1-second MP4 (black screen) so we can test upload fast.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=black:s=1280x720:r=24",
            "-t", "1",
            "-pix_fmt", "yuv420p",
            out_mp4,
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

# ---------------------------
# LTX2 model files (download at runtime)
# ---------------------------

MODELS_DIR = os.getenv("MODEL_PATH", "/models")

LTX2_FILES = [
    ("ltx-2-19b-distilled-fp8.safetensors",
     "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors"),
    ("ltx-2-spatial-upscaler-x2-1.0.safetensors",
     "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
    ("ltx-2-19b-distilled-lora-384.safetensors",
     "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"),
]

def ensure_models() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname, url in LTX2_FILES:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"⬇️ Downloading {fname}...")
            subprocess.check_call(["wget", "-O", path, url])
            print(f"✅ Downloaded {fname}")
        else:
            print(f"✅ Already have {fname}")

# ---------------------------
# LTX2 imports + pipelines (lazy load)
# ---------------------------

LTX_AVAILABLE = False
LTX_IMPORT_ERROR: Optional[str] = None

try:
    from ltx.pipelines import TI2VidTwoStagesPipeline, DistilledPipeline  # type: ignore
    LTX_AVAILABLE = True
except Exception as e1:
    try:
        from ltx_pipelines import TI2VidTwoStagesPipeline, DistilledPipeline  # type: ignore
        LTX_AVAILABLE = True
    except Exception as e2:
        LTX_IMPORT_ERROR = f"{type(e2).__name__}: {str(e2)}"
        print(f"❌ LTX2 import failed: {LTX_IMPORT_ERROR}")

pipeline = None
fast_pipeline = None

def initialize_models() -> None:
    global pipeline, fast_pipeline

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    # IMPORTANT:
    # This assumes the pipelines can load from MODEL_PATH and find weights.
    # If LTX2 expects a different layout, we'll adjust after first successful import.
    print("🧠 Initializing LTX2 pipelines...")
    pipeline = TI2VidTwoStagesPipeline.from_pretrained(
        MODELS_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    fast_pipeline = DistilledPipeline.from_pretrained(
        MODELS_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("✅ LTX2 pipelines loaded")

# ---------------------------
# LTX2 mode (placeholder until we wire real MP4 encoding)
# ---------------------------

def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function is intentionally conservative right now.

    Your previous code attempted:
      - calling the pipeline
      - then torch.save(result) into .mp4

    That will never produce a playable MP4.
    First we must confirm:
      1) models download
      2) pipelines import and load
      3) pipeline call returns a tensor/frames we can encode with ffmpeg/av

    So for now:
      - we download models
      - load pipelines
      - return a clear "ready to wire generation" response
    """

    global pipeline, fast_pipeline

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {
            "success": False,
            "mode": "ltx2",
            "error": "No prompt provided",
            "worker_version": WORKER_VERSION,
        }

    ensure_models()

    if pipeline is None or fast_pipeline is None:
        initialize_models()

    quality = (job_input.get("quality", "fast") or "fast").lower()
    selected = fast_pipeline if quality != "high" else pipeline

    return {
        "success": True,
        "mode": "ltx2",
        "message": "LTX2 pipelines loaded successfully. Next step is wiring actual generation + MP4 encoding.",
        "prompt": prompt,
        "quality": quality,
        "selected_pipeline": str(type(selected)),
        "models_dir": MODELS_DIR,
        "worker_version": WORKER_VERSION,
    }

# ---------------------------
# RunPod handler
# ---------------------------

def unwrap_input(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input") or {}
    if isinstance(job_input, dict) and isinstance(job_input.get("input"), dict):
        job_input = job_input["input"]
    return job_input

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("id", "unknown")
    print(f"📦 Processing job {job_id}")

    try:
        job_input = unwrap_input(job)
        mode = (job_input.get("mode", "smoke") or "smoke").lower()
        print(f"🧭 mode={mode}")

        if mode == "ltx2":
            result = generate_video_ltx2(job_input)
        else:
            result = generate_video_smoke(job_input)

        # Ensure version present
        if isinstance(result, dict) and "worker_version" not in result:
            result["worker_version"] = WORKER_VERSION

        return result

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "job_id": job_id,
            "worker_version": WORKER_VERSION,
        }

print("🚀 Starting RunPod serverless worker")
runpod.serverless.start({"handler": handler})
