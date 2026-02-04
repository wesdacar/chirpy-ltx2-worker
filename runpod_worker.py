"""
LTX2 Video Generation RunPod Worker (Chirpy)

Modes:
- smoke: generate 1s MP4 + upload to R2 (health check)
- ltx2 : initialize LTX2 pipelines, generate video, upload to R2

Required R2 env vars:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Recommended RunPod env vars:
- MODEL_PATH=/runpod-volume/models
- TMP_DIR=/runpod-volume/tmp
- GEMMA_ROOT=/runpod-volume/gemma
- HF_TOKEN=<huggingface token>
Optional:
- GEMMA_REPO=google/gemma-3-4b-it   (must be a repo your token can access)
"""

import os
import uuid
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List

import boto3
from botocore.config import Config
import runpod


# -----------------------
# Version + disk paths
# -----------------------
WORKER_VERSION = "v-ltx2-fixed-gemma-1"
print(f"✅ Worker booted: {WORKER_VERSION}")

def _pick_base() -> str:
    for base in [
        os.getenv("RUNPOD_VOLUME_PATH", ""),
        "/runpod-volume",
        "/workspace",
        "/volume",
        "/data",
        "/mnt",
        "/tmp",
    ]:
        if not base:
            continue
        try:
            Path(base).mkdir(parents=True, exist_ok=True)
            test = Path(base) / ".write_test"
            test.write_text("ok")
            test.unlink()
            return base
        except Exception:
            pass
    return "/tmp"

BASE = _pick_base()

# Support both names, because you used both during the last 3 days
MODEL_PATH = (
    os.getenv("MODEL_PATH")
    or os.getenv("MODEL_PATH")
    or str(Path(BASE) / "models")
)
TMP_DIR = os.getenv("TMP_DIR") or str(Path(BASE) / "tmp")
GEMMA_ROOT = os.getenv("GEMMA_ROOT") or str(Path(BASE) / "gemma")
GEMMA_REPO = os.getenv("GEMMA_REPO") or "google/gemma-3-4b-it"

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
Path(GEMMA_ROOT).mkdir(parents=True, exist_ok=True)

# Force temp usage away from /app/tmp
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR


# -----------------------
# R2 upload
# -----------------------
def upload_file_to_r2(local_path: str, content_type: str = "video/mp4") -> str:
    required = ["R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing R2 env vars: {', '.join(missing)}")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    bucket = os.environ["R2_BUCKET"]
    key = f"ltx2/{uuid.uuid4().hex}.mp4"

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
# Smoke mode
# -----------------------
def generate_video_smoke(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt", "Hello")
    out_mp4 = str(Path(TMP_DIR) / f"smoke_{uuid.uuid4().hex}.mp4")

    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=24",
         "-t", "1",
         "-pix_fmt", "yuv420p",
         out_mp4],
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
        "paths": {"model_path": MODEL_PATH, "tmp_dir": TMP_DIR, "gemma_root": GEMMA_ROOT},
        "video_url": video_url,
        "worker_version": WORKER_VERSION,
    }


# -----------------------
# LTX2 model files
# -----------------------
LTX2_FILES = [
    ("ltx-2-19b-distilled-fp8.safetensors",
     "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors"),
    ("ltx-2-spatial-upscaler-x2-1.0.safetensors",
     "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
    ("ltx-2-19b-distilled-lora-384.safetensors",
     "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"),
]

def ensure_ltx_models() -> None:
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
# Gemma download (required by your installed LTX pipelines)
# -----------------------
def ensure_gemma() -> None:
    # If folder already has content, assume it's ok
    if any(Path(GEMMA_ROOT).iterdir()):
        print(f"✅ Gemma root already populated: {GEMMA_ROOT}")
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Gemma is required by this LTX pipeline build, but HF_TOKEN is not set. "
            "Add HF_TOKEN + GEMMA_ROOT env vars in RunPod."
        )

    print(f"⬇️ Downloading Gemma repo {GEMMA_REPO} -> {GEMMA_ROOT}")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=GEMMA_ROOT,
        local_dir_use_symlinks=False,
        token=hf_token,
        # Keep it simple first; we can restrict patterns after we confirm it boots
    )

    print("✅ Gemma download complete")


# -----------------------
# Import LTX pipelines
# -----------------------
LTX_AVAILABLE = False
LTX_IMPORT_ERROR: Optional[str] = None

TI2VidTwoStagesPipeline = None
DistilledPipeline = None

try:
    from ltx_pipelines import TI2VidTwoStagesPipeline as _TI2
    from ltx_pipelines import DistilledPipeline as _DP
    TI2VidTwoStagesPipeline = _TI2
    DistilledPipeline = _DP
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

def initialize_models() -> None:
    print("🧠 Initializing LTX2 ... (DEBUG SIGNATURE MODE)")
    import inspect
    print("=== LTX2 DEBUG START ===")
    print("TI2VidTwoStagesPipeline:", TI2VidTwoStagesPipeline)
    print("DistilledPipeline:", DistilledPipeline)
    print("TI2VidTwoStagesPipeline signature:", inspect.signature(TI2VidTwoStagesPipeline))
    print("TI2VidTwoStagesPipeline.__init__:", inspect.signature(TI2VidTwoStagesPipeline.__init__))
    print("DistilledPipeline signature:", inspect.signature(DistilledPipeline))
    print("DistilledPipeline.__init__:", inspect.signature(DistilledPipeline.__init__))
    print("=== LTX2 DEBUG END ===")
    raise RuntimeError("DEBUG ONLY: printed pipeline constructor signatures")

def initialize_pipelines() -> None:
    """
    Your current runtime errors show these are REQUIRED:
      - distilled_lora
      - spatial_upsampler_path
      - gemma_root
      - loras
    """
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    if pipeline_hq is not None and pipeline_fast is not None:
        return

    print("🧠 Initializing LTX2 pipelines...")

    ensure_ltx_models()
    ensure_gemma()

    fp8_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-fp8.safetensors")
    up_path  = str(Path(MODEL_PATH) / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    lora_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-lora-384.safetensors")

    # IMPORTANT: loras is a list
    loras: List[str] = [lora_path]

    # HQ pipeline
    try:
        pipeline_hq = TI2VidTwoStagesPipeline(  # type: ignore
            fp8_path,
            lora_path,
            up_path,
            GEMMA_ROOT,
            loras,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to init HQ pipeline via positional signature: {type(e).__name__}: {e}")

    # Fast pipeline
    try:
        pipeline_fast = DistilledPipeline(  # type: ignore
            fp8_path,
            GEMMA_ROOT,
            up_path,
            loras,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to init FAST pipeline via positional signature: {type(e).__name__}: {e}")

    print(f"✅ Pipelines ready: hq={type(pipeline_hq)} fast={type(pipeline_fast)}")


def _save_to_mp4(result: Any) -> str:
    out_mp4 = str(Path(TMP_DIR) / f"ltx2_{uuid.uuid4().hex}.mp4")

    if isinstance(result, str) and Path(result).exists():
        Path(result).rename(out_mp4)
        return out_mp4

    if isinstance(result, dict):
        for k in ["video_path", "path", "mp4"]:
            vp = result.get(k)
            if isinstance(vp, str) and Path(vp).exists():
                Path(vp).rename(out_mp4)
                return out_mp4

    if hasattr(result, "save") and callable(getattr(result, "save")):
        result.save(out_mp4)  # type: ignore
        if Path(out_mp4).exists():
            return out_mp4

    raise RuntimeError(f"Unknown pipeline output type: {type(result)}")


def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "error": "No prompt provided", "worker_version": WORKER_VERSION}

    duration = int(job_input.get("duration", 1))
    fps = int(job_input.get("fps", 8))
    width = int(job_input.get("width", 512))
    height = int(job_input.get("height", 288))
    quality = (job_input.get("quality", "fast") or "fast").lower()

    num_frames = duration * fps
    steps = int(job_input.get("steps", 4 if quality == "fast" else 20))

    initialize_pipelines()

    selected = pipeline_fast if quality == "fast" else pipeline_hq
    if selected is None:
        raise RuntimeError("Selected pipeline is None")

    print(f"🎬 Generating: {duration}s {width}x{height} frames={num_frames} steps={steps} quality={quality}")
    start = time.time()

    result = selected(  # type: ignore
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=float(job_input.get("guidance_scale", 4.0)),
        enhance_prompt=bool(job_input.get("enhance_prompt", False)),
    )

    gen_time = time.time() - start
    print(f"⚡ Done in {gen_time:.2f}s")

    out_mp4 = _save_to_mp4(result)
    video_url = upload_file_to_r2(out_mp4)

    return {
        "success": True,
        "mode": "ltx2",
        "quality": quality,
        "video_url": video_url,
        "generation_time": gen_time,
        "worker_version": WORKER_VERSION,
        "paths": {"model_path": MODEL_PATH, "tmp_dir": TMP_DIR, "gemma_root": GEMMA_ROOT},
    }


# -----------------------
# RunPod handler
# -----------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    # Protect against RunPod odd jobs missing keys
    job_id = job.get("id") or job.get("requestId") or "unknown"
    print(f"📦 Processing job {job_id}")

    try:
        job_input = job.get("input") or {}
        if isinstance(job_input, dict) and "input" in job_input and isinstance(job_input["input"], dict):
            job_input = job_input["input"]

        mode = (job_input.get("mode", "smoke") or "smoke").lower()
        print(f"🧭 mode={mode}")

        if mode == "ltx2":
            initialize_models()  # TEMP DEBUG: prints pipeline signatures then stops
            return generate_video_ltx2(job_input)

        return generate_video_smoke(job_input)

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {"success": False, "job_id": job_id, "error": str(e), "worker_version": WORKER_VERSION}

if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
