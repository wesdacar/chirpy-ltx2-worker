"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: download models + gemma (if missing), init pipelines, generate video, upload to R2

Required env vars for R2:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Required for LTX2 (Gemma):
- HF_TOKEN=xxxxxxxxxxxxxxxx
- GEMMA_ROOT=/runpod-volume/gemma   (recommended)

Recommended on RunPod:
- MODEL_PATH=/runpod-volume/models
- TMP_DIR=/runpod-volume/tmp
"""

import os
import uuid
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
import inspect

import boto3
from botocore.config import Config
import runpod


# -----------------------
# Version + basic config
# -----------------------
WORKER_VERSION = "v-ltx2-final-1"
print(f"✅ Worker booted: {WORKER_VERSION}")


# -----------------------
# Pick a writable big disk
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
            p = Path(base)
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink()
            return str(p)
        except Exception:
            continue
    return "/tmp"


BASE = pick_writable_base()

# Use env if provided, otherwise default to big disk
MODEL_PATH = os.getenv("MODEL_PATH") or str(Path(BASE) / "models")
TMP_DIR = os.getenv("TMP_DIR") or str(Path(BASE) / "tmp")
GEMMA_ROOT = os.getenv("GEMMA_ROOT") or str(Path(BASE) / "gemma")

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
Path(GEMMA_ROOT).mkdir(parents=True, exist_ok=True)

# Force Python/temp usage onto big disk (prevents /app/tmp filling)
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR


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
        "paths": {
            "base": BASE,
            "model_path": MODEL_PATH,
            "tmp_dir": TMP_DIR,
            "gemma_root": GEMMA_ROOT,
        },
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
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    for fname, url in LTX2_FILES:
        dst = Path(MODEL_PATH) / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"✅ Already have {fname}")
            continue

        print(f"⬇️ Downloading {fname} -> {dst}")
        subprocess.check_call(["wget", "-c", "-O", str(dst), url])
        print(f"✅ Downloaded {fname}")


# -----------------------
# Gemma download (required by your constructor signature)
# -----------------------
# Choose a repo default (you can override with GEMMA_REPO env)
GEMMA_REPO = os.getenv("GEMMA_REPO") or "google/gemma-3-12b-it-qat-q4_0-unquantized"


def ensure_gemma() -> None:
    """
    Downloads Gemma repo into GEMMA_ROOT using HF_TOKEN.
    This must succeed or pipelines can't init.
    """
    Path(GEMMA_ROOT).mkdir(parents=True, exist_ok=True)

    # quick check: if folder has files, assume it's already present
    if any(Path(GEMMA_ROOT).glob("**/*")):
        print(f"✅ Gemma already present in {GEMMA_ROOT}")
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN env var (required to download Gemma).")

    print(f"⬇️ Downloading Gemma repo {GEMMA_REPO} -> {GEMMA_ROOT}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not available in image: {e}")

    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=GEMMA_ROOT,
        local_dir_use_symlinks=False,
        token=token,
        # don’t filter; let HF decide best files for the repo
    )

    print("✅ Gemma download complete")


# -----------------------
# Import LTX pipelines + lora primitive
# -----------------------
LTX_AVAILABLE = False
LTX_IMPORT_ERROR: Optional[str] = None

TI2VidTwoStagesPipeline = None
DistilledPipeline = None
LoraPathStrengthAndSDOps = None

try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline as _TI2
    from ltx_pipelines.distilled import DistilledPipeline as _DP
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps as _LORA

    TI2VidTwoStagesPipeline = _TI2
    DistilledPipeline = _DP
    LoraPathStrengthAndSDOps = _LORA

    LTX_AVAILABLE = True
    print("✅ Imported ltx_pipelines + ltx_core primitives")
except Exception as e:
    LTX_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"❌ LTX import failed: {LTX_IMPORT_ERROR}")


# -----------------------
# Build LoraPathStrengthAndSDOps safely (signature varies sometimes)
# -----------------------
def make_lora(path: str, strength: float = 1.0) -> Any:
    if LoraPathStrengthAndSDOps is None:
        raise RuntimeError("LoraPathStrengthAndSDOps not importable")

    sig = inspect.signature(LoraPathStrengthAndSDOps)
    params = list(sig.parameters.keys())

    # Common patterns:
    # (path, strength, sd_ops) OR (path, strength) OR keyword fields
    try:
        if len(params) >= 2:
            return LoraPathStrengthAndSDOps(path, strength)  # type: ignore
    except TypeError:
        pass

    try:
        return LoraPathStrengthAndSDOps(path=path, strength=strength)  # type: ignore
    except TypeError:
        pass

    # last resort: just pass path
    return LoraPathStrengthAndSDOps(path)  # type: ignore


# -----------------------
# Pipelines (lazy loaded)
# -----------------------
pipeline_hq = None
pipeline_fast = None


def initialize_models() -> None:
    """
    Uses the EXACT signatures you printed:

    TI2VidTwoStagesPipeline(
        checkpoint_path,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path,
        gemma_root,
        loras: list[LoraPathStrengthAndSDOps],
        ...
    )

    DistilledPipeline(
        checkpoint_path,
        gemma_root,
        spatial_upsampler_path,
        loras: list[LoraPathStrengthAndSDOps],
        ...
    )
    """
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    if pipeline_hq is not None and pipeline_fast is not None:
        return

    print("🧠 Initializing LTX2 pipelines... (signature-confirmed)")

    ensure_models()
    ensure_gemma()

    fp8_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-fp8.safetensors")
    up_path = str(Path(MODEL_PATH) / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    distilled_lora_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-lora-384.safetensors")

    distilled_lora_list = [make_lora(distilled_lora_path, 1.0)]
    loras_list: List[Any] = []  # keep empty for now

    # HQ
    pipeline_hq = TI2VidTwoStagesPipeline(  # type: ignore
        fp8_path,
        distilled_lora_list,
        up_path,
        GEMMA_ROOT,
        loras_list,
    )

    # FAST
    pipeline_fast = DistilledPipeline(  # type: ignore
        fp8_path,
        GEMMA_ROOT,
        up_path,
        loras_list,
    )

    print(f"✅ Pipelines loaded: hq={type(pipeline_hq)} fast={type(pipeline_fast)}")


def _save_pipeline_output_to_mp4(result: Any) -> str:
    """
    Minimal saver.
    If output type differs, we’ll adjust based on what we see.
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

    # If we get here, we need to see what result is
    raise RuntimeError(
        f"Unknown pipeline output type: {type(result)}. "
        f"Keys={list(result.keys()) if isinstance(result, dict) else 'n/a'}"
    )


def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small-first text-to-video implementation.
    Start small so we know it works, then scale.
    """
    global pipeline_hq, pipeline_fast

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "error": "No prompt provided", "worker_version": WORKER_VERSION}

    quality = (job_input.get("quality", "fast") or "fast").lower()

    # Safe defaults: keep small unless you override
    duration = int(job_input.get("duration", 2))
    fps = int(job_input.get("fps", 12))
    width = int(job_input.get("width", 512))
    height = int(job_input.get("height", 288))
    steps = int(job_input.get("steps", 6 if quality == "fast" else 20))

    num_frames = duration * fps
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    enhance_prompt = bool(job_input.get("enhance_prompt", True))

    if pipeline_hq is None or pipeline_fast is None:
        initialize_models()

    selected = pipeline_fast if quality == "fast" else pipeline_hq
    if selected is None:
        raise RuntimeError("Selected pipeline is None after initialize_models()")

    print(f"🎬 LTX2 generating: {duration}s {width}x{height} fps={fps} frames={num_frames} steps={steps} quality={quality}")
    start = time.time()

    # Call pipeline: keep args compatible
    try:
        result = selected(  # type: ignore
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            enhance_prompt=enhance_prompt,
        )
    except TypeError:
        # fallback: minimal call
        result = selected(prompt=prompt)  # type: ignore

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
        "paths": {
            "base": BASE,
            "model_path": MODEL_PATH,
            "tmp_dir": TMP_DIR,
            "gemma_root": GEMMA_ROOT,
        },
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
            "gemma_repo": GEMMA_REPO,
        },
    }


# -----------------------
# RunPod handler
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

        if mode == "ltx2":
            return generate_video_ltx2(job_input)
        return generate_video_smoke(job_input)

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "error": str(e),
            "worker_version": WORKER_VERSION,
            "paths": {
                "base": BASE,
                "model_path": MODEL_PATH,
                "tmp_dir": TMP_DIR,
                "gemma_root": GEMMA_ROOT,
            },
        }


# Only start worker when executed directly (prevents Dockerfile import checks from starting it)
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
