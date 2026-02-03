"""
LTX2 Video Generation RunPod Worker (RunPod Serverless)

Modes:
  - smoke: generates a 1s black MP4 and uploads to Cloudflare R2
  - ltx2:
      task="t2v" => text-to-video
      task="i2v" => image-to-video (requires image_url)

Notes:
  - Models are downloaded at runtime to /models (MODEL_PATH)
  - Pipelines are lazy-loaded once per worker
  - Frames are encoded to MP4 with ffmpeg
  - The server does NOT auto-start on import (Docker build import-check safe)
"""

import os
import uuid
import time
import shutil
import inspect
import subprocess
from typing import Any, Dict, Optional, List

import boto3
from botocore.config import Config
import runpod
import torch

import numpy as np
from PIL import Image


# ---------------------------
# Version
# ---------------------------
WORKER_VERSION = "v-ltx2-full-1"
print(f"✅ Worker booted: {WORKER_VERSION}")


# ---------------------------
# R2 upload (S3-compatible)
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
# Input helpers
# ---------------------------
def unwrap_input(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input") or {}
    # RunPod UI sometimes double-wraps
    if isinstance(job_input, dict) and isinstance(job_input.get("input"), dict):
        job_input = job_input["input"]
    return job_input


# ---------------------------
# Smoke mode (fast sanity check)
# ---------------------------
def generate_video_smoke(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt", "Hello")
    out_mp4 = "/tmp/ltx2_smoke_test.mp4"

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
# LTX2 models (download at runtime)
# ---------------------------
MODELS_DIR = os.getenv("MODEL_PATH", "/workspace/models")

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
# Image download/load for i2v
# ---------------------------
def download_to_file(url: str, out_path: str) -> None:
    subprocess.check_call(["curl", "-L", "-o", out_path, url])

def load_init_image(job_input: Dict[str, Any]) -> Optional[Image.Image]:
    image_url = (job_input.get("image_url") or "").strip()
    if not image_url:
        return None
    tmp = f"/tmp/init_{uuid.uuid4().hex}.png"
    download_to_file(image_url, tmp)
    return Image.open(tmp).convert("RGB")


# ---------------------------
# Call pipeline safely (filter kwargs to what the pipeline supports)
# ---------------------------
def call_pipeline_compat(pipe, **kwargs):
    sig = inspect.signature(pipe.__call__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return pipe(**filtered)


# ---------------------------
# Frames -> MP4 using ffmpeg
# ---------------------------
def normalize_frames(frames: Any) -> List[np.ndarray]:
    # unwrap dict outputs
    if isinstance(frames, dict):
        for k in ("frames", "images", "videos", "video"):
            if k in frames:
                frames = frames[k]
                break

    # list of PIL
    if isinstance(frames, list) and frames and isinstance(frames[0], Image.Image):
        return [np.array(im.convert("RGB"), dtype=np.uint8) for im in frames]

    # numpy
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported numpy frames shape: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]

    # torch tensor
    if torch.is_tensor(frames):
        t = frames
        if t.ndim == 5:
            t = t[0]
        # T,C,H,W -> T,H,W,C
        if t.ndim == 4 and t.shape[1] in (1, 3, 4):
            t = t.permute(0, 2, 3, 1)
        if t.ndim != 4:
            raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")

        t = t.detach().float().cpu()

        # heuristic scaling
        if t.min() < 0:
            t = (t + 1.0) / 2.0
        if t.max() <= 1.5:
            t = t * 255.0

        t = t.clamp(0, 255).byte().numpy()
        return [t[i] for i in range(t.shape[0])]

    raise ValueError(f"Unsupported frames type: {type(frames)}")

def frames_to_mp4(frames: Any, fps: int, out_mp4: str) -> None:
    tmp_dir = f"/tmp/frames_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)

    imgs = normalize_frames(frames)

    for i, frame in enumerate(imgs):
        Image.fromarray(frame, mode="RGB").save(os.path.join(tmp_dir, f"frame_{i:05d}.png"))

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------
# LTX2 generation (t2v + i2v)
# ---------------------------
def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    global pipeline, fast_pipeline

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "mode": "ltx2", "error": "No prompt provided", "worker_version": WORKER_VERSION}

    task = (job_input.get("task") or "t2v").lower()
    quality = (job_input.get("quality") or "fast").lower()

    duration = int(job_input.get("duration", 2))
    fps = int(job_input.get("fps", 12))
    width = int(job_input.get("width", 768))
    height = int(job_input.get("height", 432))
    num_frames = max(1, duration * fps)

    ensure_models()

    if pipeline is None or fast_pipeline is None:
        initialize_models()

    pipe = fast_pipeline if quality != "high" else pipeline

    init_image = None
    if task == "i2v":
        init_image = load_init_image(job_input)
        if init_image is None:
            return {
                "success": False,
                "mode": "ltx2",
                "task": "i2v",
                "error": "task=i2v requires image_url",
                "worker_version": WORKER_VERSION,
            }

    print(f"🎬 LTX2 {task} | {width}x{height} | frames={num_frames} | quality={quality}")

    start = time.time()

    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": 40 if quality == "high" else 8,
        "guidance_scale": float(job_input.get("guidance_scale", 7.5)),
        "enhance_prompt": bool(job_input.get("enhance_prompt", True)),
        # common image args
        "image": init_image,
        "init_image": init_image,
        "conditioning_image": init_image,
    }

    result = call_pipeline_compat(pipe, **kwargs)

    gen_s = time.time() - start
    print(f"⚡ Pipeline finished in {gen_s:.2f}s")

    # Extract frames from common output patterns
    frames = result
    if isinstance(result, dict):
        for k in ("frames", "images", "videos", "video"):
            if k in result:
                frames = result[k]
                break
    else:
        if hasattr(result, "frames"):
            frames = getattr(result, "frames")
        elif hasattr(result, "images"):
            frames = getattr(result, "images")
        elif hasattr(result, "videos"):
            frames = getattr(result, "videos")

    out_mp4 = f"/tmp/ltx2_{uuid.uuid4().hex}.mp4"
    frames_to_mp4(frames, fps=fps, out_mp4=out_mp4)

    video_url = upload_file_to_r2(out_mp4)

    return {
        "success": True,
        "mode": "ltx2",
        "task": task,
        "prompt": prompt,
        "video_url": video_url,
        "generation_time_s": round(gen_s, 3),
        "settings": {
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "quality": quality,
        },
        "worker_version": WORKER_VERSION,
    }


# ---------------------------
# RunPod handler
# ---------------------------
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

        if isinstance(result, dict) and "worker_version" not in result:
            result["worker_version"] = WORKER_VERSION

        return result

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {"success": False, "error": str(e), "job_id": job_id, "worker_version": WORKER_VERSION}


# IMPORTANT:
# Guard so Docker build import-check does NOT start the RunPod worker.
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
