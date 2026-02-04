"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: (next step) generate video via LTX2 then upload to R2

Required env vars for R2:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Optional env vars:
- MODEL_PATH (we will auto-fix if it is wrongly '/models')
- TMP_DIR
- GEMMA_ROOT
- HF_TOKEN (for Gemma downloads if/when needed)
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


# -----------------------
# Base disk + path selection
# -----------------------
def pick_writable_base() -> str:
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

# Read raw env values (for debugging)
ENV_MODEL_PATH = os.getenv("MODEL_PATH")
ENV_TMP_DIR = os.getenv("TMP_DIR")
ENV_GEMMA_ROOT = os.getenv("GEMMA_ROOT")

# Default everything onto the big disk base
MODEL_PATH = ENV_MODEL_PATH or str(Path(BASE) / "models")
TMP_DIR = ENV_TMP_DIR or str(Path(BASE) / "tmp")
GEMMA_ROOT = ENV_GEMMA_ROOT or str(Path(BASE) / "gemma")

# HARD FIX: if MODEL_PATH is "/models" (wrong/small), force it onto big disk
if MODEL_PATH.strip() == "/models" and Path("/runpod-volume").exists():
    MODEL_PATH = "/runpod-volume/models"

# Create dirs
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
Path(GEMMA_ROOT).mkdir(parents=True, exist_ok=True)

# Force temp usage off /app/tmp
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

# Also export corrected vars so the rest of the process sees them
os.environ["MODEL_PATH"] = MODEL_PATH
os.environ["TMP_DIR"] = TMP_DIR
os.environ["GEMMA_ROOT"] = GEMMA_ROOT


# -----------------------
# Version
# -----------------------
WORKER_VERSION = "v-ltx2-final-2"
print(f"✅ Worker booted: {WORKER_VERSION}")
print(f"🧱 BASE={BASE}")
print(f"📦 MODEL_PATH={MODEL_PATH}")
print(f"🗂️ TMP_DIR={TMP_DIR}")
print(f"🧠 GEMMA_ROOT={GEMMA_ROOT}")


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
        ExpiresIn=60 * 60,
    )


# -----------------------
# Smoke test
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
        "paths": {
            "base": BASE,
            "model_path": MODEL_PATH,
            "tmp_dir": TMP_DIR,
            "gemma_root": GEMMA_ROOT,
        },
        "raw_env": {
            "MODEL_PATH": ENV_MODEL_PATH,
            "TMP_DIR": ENV_TMP_DIR,
            "GEMMA_ROOT": ENV_GEMMA_ROOT,
        },
    }


# -----------------------
# Handler
# -----------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("id") or job.get("requestId") or "unknown"
    print(f"📦 Processing job {job_id}")

    try:
        job_input = job.get("input") or {}
        if isinstance(job_input, dict) and "input" in job_input and isinstance(job_input["input"], dict):
            job_input = job_input["input"]

        mode = (job_input.get("mode", "smoke") or "smoke").lower()
        print(f"🧭 mode={mode}")

        # For now keep LTX2 disabled until paths are correct
        if mode == "ltx2":
            return {
                "success": False,
                "error": "LTX2 temporarily disabled until model_path is confirmed correct",
                "worker_version": WORKER_VERSION,
                "paths": {"base": BASE, "model_path": MODEL_PATH, "tmp_dir": TMP_DIR, "gemma_root": GEMMA_ROOT},
                "raw_env": {"MODEL_PATH": ENV_MODEL_PATH, "TMP_DIR": ENV_TMP_DIR, "GEMMA_ROOT": ENV_GEMMA_ROOT},
            }

        return generate_video_smoke(job_input)

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {"success": False, "job_id": job_id, "error": str(e), "worker_version": WORKER_VERSION}


if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
