"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation

Modes:
- smoke: generate a 1s black MP4 and upload to R2 (fast health check)
- ltx2: download LTX-2 weights (if missing), init pipelines, generate video, upload to R2

Required env vars for R2:
- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Recommended env vars on RunPod (good defaults if you set them):
- MODEL_PATH=/runpod-volume/models
- TMP_DIR=/runpod-volume/tmp
- TMPDIR=/runpod-volume/tmp

Optional (ONLY if you want HQ pipeline later):
- GEMMA_ROOT=/runpod-volume/gemma   (must exist + contain Gemma files)
"""

import os
import uuid
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import boto3
from botocore.config import Config
import runpod


# -----------------------
# Version
# -----------------------
WORKER_VERSION = "v-ltx2-clean-full-1"
print(f"✅ Worker booted: {WORKER_VERSION}")


# -----------------------
# Disk / tmp selection
# -----------------------
def pick_writable_base() -> str:
    """
    Finds a writable base path with space.
    Priority: RunPod volume mounts first.
    """
    candidates = [
        os.getenv("RUNPOD_VOLUME_PATH", "").strip(),
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

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(BASE) / "models"))
TMP_DIR = os.getenv("TMP_DIR", str(Path(BASE) / "tmp"))

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

# Force Python/tempfile/libs to use big disk, not /app/tmp
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

print(f"📁 BASE={BASE}")
print(f"📁 MODEL_PATH={MODEL_PATH}")
print(f"📁 TMP_DIR={TMP_DIR}")


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
        "video_url": video_url,
        "worker_version": WORKER_VERSION,
        "paths": {"model_path": MODEL_PATH, "tmp_dir": TMP_DIR},
    }


# -----------------------
# LTX-2 weights
# -----------------------
LTX2_FILES: List[Tuple[str, str]] = [
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
            print(f"✅ Already have {fname} ({dst.stat().st_size} bytes)")
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
    # Your container installs ltx-pipelines and imports as ltx_pipelines
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
# Pipelines (lazy)
# -----------------------
pipeline_hq = None
pipeline_fast = None


def initialize_models() -> None:
    """
    Initializes FAST pipeline first (no Gemma requirement).
    HQ pipeline is optional and only loads if GEMMA_ROOT exists.
    """
    global pipeline_hq, pipeline_fast

    if not LTX_AVAILABLE:
        raise RuntimeError(f"LTX2 packages not available. Import error: {LTX_IMPORT_ERROR}")

    if pipeline_fast is not None or pipeline_hq is not None:
        # If any exists, we don't force reload; that's enough to proceed.
        return

    print("🧠 Initializing LTX2 pipelines... (constructor API)")
    ensure_models()

    fp8_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-fp8.safetensors")
    upsampler_path = str(Path(MODEL_PATH) / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    distilled_lora_path = str(Path(MODEL_PATH) / "ltx-2-19b-distilled-lora-384.safetensors")

    # ---- FAST pipeline (try a few constructor shapes)
    print("⚡ Initializing FAST pipeline...")
    fast_attempts: List[str] = []

    for label, pos_args, kw_args in [
        ("pos_fp8", [fp8_path], {}),
        ("pos_fp8_lora", [fp8_path, distilled_lora_path], {}),
        ("pos_fp8_lora_upsampler", [fp8_path, distilled_lora_path, upsampler_path], {}),
    ]:
        try:
            pipeline_fast = DistilledPipeline(*pos_args, **kw_args)  # type: ignore
            print(f"✅ FAST pipeline ready via {label}: {type(pipeline_fast)}")
            break
        except Exception as e:
            fast_attempts.append(f"{label}: {type(e).__name__}: {e}")
            pipeline_fast = None

    if pipeline_fast is None:
        raise RuntimeError("Failed to init FAST pipeline. Attempts:\n" + "\n".join(fast_attempts))

    # ---- HQ pipeline optional (needs Gemma + loras signature)
    gemma_root = (os.getenv("GEMMA_ROOT") or "").strip()
    if not gemma_root:
        print("ℹ️ GEMMA_ROOT not set. Skipping HQ pipeline init (FAST will work).")
        return

    if not Path(gemma_root).exists():
        print(f"ℹ️ GEMMA_ROOT does not exist: {gemma_root}. Skipping HQ pipeline init.")
        return

    print(f"🎯 Initializing HQ pipeline using GEMMA_ROOT={gemma_root}")
    try:
        # Your logs indicated required args:
        # (fp8_path, distilled_lora, spatial_upsampler_path, gemma_root, loras)
        pipeline_hq = TI2VidTwoStagesPipeline(  # type: ignore
            fp8_path,
            distilled_lora_path,
            upsampler_path,
            gemma_root,
            [],  # loras list
        )
        print(f"✅ HQ pipeline ready: {type(pipeline_hq)}")
    except Exception as e:
        print(f"⚠️ HQ pipeline init failed: {type(e).__name__}: {e}")
        pipeline_hq = None


def _save_pipeline_output_to_mp4(result: Any) -> str:
    """
    Best-effort saver. We will adjust once we see real output type.
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

    raise RuntimeError(f"Unknown pipeline output type: {type(result)} (no save/video_path/path/mp4)")


def generate_video_ltx2(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Text-to-video.
    For now, we default to FAST because HQ requires Gemma.
    If you later add GEMMA_ROOT, you can request quality=high.
    """
    global pipeline_hq, pipeline_fast

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"success": False, "error": "No prompt provided", "worker_version": WORKER_VERSION}

    duration = int(job_input.get("duration", 2))
    fps = int(job_input.get("fps", 16))
    width = int(job_input.get("width", 768))
    height = int(job_input.get("height", 432))
    quality = (job_input.get("quality", "fast") or "fast").lower()

    num_frames = duration * fps
    steps = int(job_input.get("steps", 8 if quality != "high" else 30))
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    enhance_prompt = bool(job_input.get("enhance_prompt", True))

    # Ensure weights exist + pipelines loaded
    initialize_models()

    # HQ only if present; otherwise fall back to fast
    selected = pipeline_hq if (quality == "high" and pipeline_hq is not None) else pipeline_fast
    if selected is None:
        raise RuntimeError("No pipeline available after initialize_models()")

    print(
        f"🎬 LTX2 generating: {duration}s {width}x{height} fps={fps} frames={num_frames} "
        f"steps={steps} quality={quality} pipeline={type(selected)}"
    )

    start = time.time()

    # Pipeline call signature might differ; we pass common kwargs.
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
        "paths": {"model_path": MODEL_PATH, "tmp_dir": TMP_DIR},
    }


# -----------------------
# RunPod handler
# -----------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    # RunPod sometimes calls with malformed job - guard it
    if not isinstance(job, dict) or "input" not in job:
        return {"success": False, "error": "Job has missing field(s): id or input.", "worker_version": WORKER_VERSION}

    job_id = job.get("id", "unknown")
    print(f"📦 Processing job {job_id}")

    try:
        job_input = job.get("input") or {}
        if isinstance(job_input, dict) and "input" in job_input and isinstance(job_input["input"], dict):
            job_input = job_input["input"]

        mode = (job_input.get("mode", "smoke") or "smoke").lower()
        print(f"🧭 mode={mode}")

        if mode == "ltx2":
            result = generate_video_ltx2(job_input)
        else:
            result = generate_video_smoke(job_input)

        if isinstance(result, dict):
            result["worker_version"] = WORKER_VERSION
        return result

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "error": str(e),
            "worker_version": WORKER_VERSION,
        }


# IMPORTANT: only start when run as main (prevents Dockerfile import sanity checks from launching serverless)
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
