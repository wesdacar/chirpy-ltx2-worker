"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: download LTX-2 weights + Gemma, init pipelines, generate video, upload to R2

Required env vars for R2:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Required env vars for LTX2:
- MODEL_PATH=/runpod-volume/models
- TMP_DIR=/runpod-volume/tmp
- GEMMA_ROOT=/runpod-volume/gemma
- HF_TOKEN=... (Hugging Face token)

Optional:
- GEMMA_REPO (defaults below)
"""

import os
import uuid
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
import runpod


# -----------------------
# Version + paths
# -----------------------
WORKER_VERSION = "v-ltx2-real-1"
print(f"✅ Worker booted: {WORKER_VERSION}")

MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/models")
TMP_DIR = os.getenv("TMP_DIR", "/runpod-volume/tmp")
GEMMA_ROOT = os.getenv("GEMMA_ROOT", "/runpod-volume/gemma")
HF_TOKEN = os.getenv("HF_TOKEN")

# You can change this later if you want a different Gemma variant
GEMMA_REPO = os.getenv("GEMMA_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
Path(GEMMA_ROOT).mkdir(parents=True, exist_ok=True)

# Force tempfile usage onto big disk
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
        "paths": {
            "model_path": MODEL_PATH,
            "tmp_dir": TMP_DIR,
            "gemma_root": GEMMA_ROOT,
        },
        "raw_env": {
            "MODEL_PATH": os.getenv("MODEL_PATH"),
            "TMP_DIR": os.getenv("TMP_DIR"),
            "GEMMA_ROOT": os.getenv("GEMMA_ROOT"),
        },
        "video_url": video_url,
        "worker_version": WORKER_VERSION,
    }


# -----------------------
# LTX2 weights download
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
    for fname, url in LTX2_FILES:
        dst = Path(MODEL_PATH) / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"✅ Already have {fname}")
            continue
        print(f"⬇️ Downloading {fname} -> {dst}")
        subprocess.check_call(["wget", "-c", "-O", str(dst), url])
        print(f"✅ Downloaded {fname}")


# -----------------------
# Gemma download (HF)
# -----------------------
def ensure_gemma() -> str:
    """
    Downloads Gemma repo into GEMMA_ROOT/<repo_name>.
    Returns the local folder path that we pass to pipelines as gemma_root.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing (required to download Gemma). Add HF_TOKEN in RunPod env vars.")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not available in image. Error: {e}")

    local_dir = str(Path(GEMMA_ROOT) / GEMMA_REPO.replace("/", "__"))
    marker = Path(local_dir) / ".complete"

    if marker.exists():
        print(f"✅ Gemma already present: {local_dir}")
        return local_dir

    print(f"⬇️ Downloading Gemma repo {GEMMA_REPO} -> {local_dir}")
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    marker.write_text("ok")
    print(f"✅ Gemma download complete: {local_dir}")
    return local_dir


# -----------------------
# Import + pipeline init
# -----------------------
LTX_AVAILABLE = False
LTX_IMPORT_ERROR: Optional[str] = None

TI2VidTwoStagesPipeline = None
DistilledPipeline = None

try:
    from ltx_pipelines import TI2VidTwoStagesPipeline as _TI2
    from ltx_pipelines import DistilledPipeline as _DP
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps  # type: ignore

    TI2VidTwoStagesPipeline = _TI2
    DistilledPipeline = _DP
    LTX_AVAILABLE = True
    print("✅ Imported ltx_pipelines + LoraPathStrengthAndSDOps")
except Exception as e:
    LTX_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"❌ LTX import failed: {LTX_IMPORT_ERROR}")

pipeline_hq = None
pipeline_fast = None


def _mk_lora_list(lora_path: str) -> List[Any]:
    """
    Build the required list[ LoraPathStrengthAndSDOps ].
    We default strength=1.0 and sd_ops="none" (safe default).
    """
    # Some builds accept sd_ops as a string; others as enum.
    # We'll try simplest construction first.
    try:
        return [LoraPathStrengthAndSDOps(path=lora_path, strength=1.0, sd_ops="none")]
    except TypeError:
        # Alternate param names
        return [LoraPathStrengthAndSDOps(lora_path, 1.0, "none")]


def initialize_models() -> None:
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    if pipeline_hq is not None and pipeline_fast is not None:
        return

    print("🧠 Initializing LTX2 pipelines...")

    ensure_models()
    gemma_local = ensure_gemma()

    ckpt = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-fp8.safetensors")
    upsampler = str(Path(MODEL_PATH) / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    distilled_lora_file = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-lora-384.safetensors")

    distilled_lora = _mk_lora_list(distilled_lora_file)
    loras = _mk_lora_list(distilled_lora_file)  # start with same lora list (safe minimal)

    # Constructor signature we confirmed:
    # TI2VidTwoStagesPipeline(checkpoint_path, distilled_lora, spatial_upsampler_path, gemma_root, loras, device=..., fp8transformer=False)
    pipeline_hq = TI2VidTwoStagesPipeline(  # type: ignore
        ckpt,
        distilled_lora,
        upsampler,
        gemma_local,
        loras,
    )

    # DistilledPipeline(checkpoint_path, gemma_root, spatial_upsampler_path, loras, device=..., fp8transformer=False)
    pipeline_fast = DistilledPipeline(  # type: ignore
        ckpt,
        gemma_local,
        upsampler,
        loras,
    )

    print(f"✅ Pipelines ready: hq={type(pipeline_hq)} fast={type(pipeline_fast)}")


def _save_output_to_mp4(result: Any) -> str:
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

    raise RuntimeError(f"Unknown pipeline output type: {type(result)} (no save/path/video_path)")


def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    global pipeline_hq, pipeline_fast

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "error": "No prompt provided", "worker_version": WORKER_VERSION}

    duration = int(job_input.get("duration", 5))
    fps = int(job_input.get("fps", 24))
    width = int(job_input.get("width", 1280))
    height = int(job_input.get("height", 720))
    quality = (job_input.get("quality", "fast") or "fast").lower()

    num_frames = duration * fps
    steps = int(job_input.get("steps", 8 if quality == "fast" else 30))

    if pipeline_hq is None or pipeline_fast is None:
        initialize_models()

    selected = pipeline_fast if quality == "fast" else pipeline_hq
    if selected is None:
        raise RuntimeError("Selected pipeline is None after initialize_models()")

    print(f"🎬 LTX2 generate: quality={quality} {duration}s {width}x{height} fps={fps} frames={num_frames} steps={steps}")
    start = time.time()

    result = selected(  # type: ignore
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=float(job_input.get("guidance_scale", 7.5)),
        enhance_prompt=bool(job_input.get("enhance_prompt", True)),
    )

    gen_time = time.time() - start
    print(f"⚡ Generation finished in {gen_time:.2f}s")

    out_mp4 = _save_output_to_mp4(result)
    video_url = upload_file_to_r2(out_mp4)

    return {
        "success": True,
        "mode": "ltx2",
        "quality": quality,
        "video_url": video_url,
        "generation_time": gen_time,
        "worker_version": WORKER_VERSION,
        "paths": {
            "model_path": MODEL_PATH,
            "tmp_dir": TMP_DIR,
            "gemma_root": GEMMA_ROOT,
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
        }


if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
