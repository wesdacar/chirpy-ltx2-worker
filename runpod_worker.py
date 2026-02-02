"""
LTX2 Video Generation RunPod Worker
Serverless endpoint for Chirpy.me video generation
"""

import os
import uuid
import subprocess
import boto3
from botocore.config import Config
import runpod
import torch
import os
import time
import requests
from io import BytesIO
from pathlib import Path
import tempfile
import uuid
import boto3
from botocore.exceptions import NoCredentialsError

# Import LTX2 pipeline (will be available after model installation)
try:
    from ltx_pipelines import TI2VidTwoStagesPipeline, DistilledPipeline
    from ltx_core.models import LTXVideoTransformer
    LTX_AVAILABLE = True
except ImportError:
    LTX_AVAILABLE = False
    print("LTX2 not available - worker will fail")

# Global variables for model loading
pipeline = None
fast_pipeline = None

def initialize_models():
    """Initialize LTX2 models - called once when worker starts"""
    global pipeline, fast_pipeline
    
    if not LTX_AVAILABLE:
        raise RuntimeError("LTX2 packages not available")
    
    model_path = os.getenv("MODEL_PATH", "/models")
    
    # Load main two-stage pipeline for high quality
    pipeline = TI2VidTwoStagesPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load fast pipeline for quick previews
    fast_pipeline = DistilledPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print("✅ LTX2 models loaded successfully")

def upload_to_s3(video_path, bucket_name, s3_key):
    """Upload generated video to S3 and return public URL"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Upload file
        s3_client.upload_file(
            video_path, 
            bucket_name, 
            s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        # Return public URL
        return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
    except Exception as e:
        print(f"S3 upload failed: {e}")
        return None

def generate_video_ltx2(job):
    """Main video generation handler"""
    try:
        # Extract parameters from job input
        job_input = job.get('input', {}) or {}
        if isinstance(job_input, dict) and "input" in job_input and isinstance(job_input["input"], dict):
            job_input = job_input["input"]
        prompt = job_input.get('prompt', '')
        
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Generation parameters
        duration = job_input.get('duration', 5)  # seconds
        width = job_input.get('width', 1280)
        height = job_input.get('height', 720)
        fps = job_input.get('fps', 24)
        quality = job_input.get('quality', 'high')  # 'high' or 'fast'
        
        # Calculate frames
        num_frames = duration * fps
        
        # Select pipeline based on quality preference
        selected_pipeline = pipeline if quality == 'high' else fast_pipeline
        
        print(f"🎬 Generating {duration}s video: {prompt[:50]}...")
        start_time = time.time()
        
        # Generate video
        result = selected_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=40 if quality == 'high' else 8,
            guidance_scale=7.5,
            enhance_prompt=True  # Use built-in prompt enhancement
        )
        
        generation_time = time.time() - start_time
        print(f"⚡ Generated in {generation_time:.2f}s")
        
        # Save video to temporary file
        video_id = str(uuid.uuid4())
        temp_path = f"/tmp/video_{video_id}.mp4"
        
        # Save the video (result contains video tensor)
        # Note: Actual saving method depends on LTX2 output format
        # This is a placeholder - check LTX2 docs for correct method
        if hasattr(result, 'save'):
            result.save(temp_path)
        else:
            # Convert tensor to video file
            # Implementation depends on LTX2 output format
            torch.save(result, temp_path)
        
        # Upload to S3 if configured
        video_url = None
        bucket_name = os.getenv('S3_BUCKET_NAME')
        
        if bucket_name:
            s3_key = f"generated/{video_id}.mp4"
            video_url = upload_to_s3(temp_path, bucket_name, s3_key)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # Return result
        return {
            "success": True,
            "video_url": video_url,
            "video_id": video_id,
            "generation_time": generation_time,
            "parameters": {
                "prompt": prompt,
                "duration": duration,
                "width": width, 
                "height": height,
                "fps": fps,
                "quality": quality
            },
            "metadata": {
                "model": "LTX2",
                "inference_steps": 40 if quality == 'high' else 8,
                "frames_generated": num_frames
            }
        }
        
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def upload_file_to_r2(local_path: str, content_type: str = "video/mp4") -> str:
    """
    Uploads a file to Cloudflare R2 (S3-compatible) and returns a presigned URL.
    (You said no public base URL yet, so presigned is perfect.)
    """
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

def generate_video_smoke(job):
    prompt = (job.get("input") or {}).get("prompt", "Hello")
    out_mp4 = "/tmp/ltx2_smoke_test.mp4"

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
        "message": "R2 upload smoke test succeeded",
        "echo": prompt,
        "video_url": video_url
    }

def handler(job):
    """RunPod handler function"""
    job_id = job.get("id", "unknown")
    print(f"📝 Processing job {job_id}")

    try:
        job_input = job.get("input", {}) or {}

        # If RunPod UI double-wraps input, unwrap one level
        if isinstance(job_input, dict) and "input" in job_input and isinstance(job_input["input"], dict):
            job_input = job_input["input"]

        mode = (job_input.get("mode", "smoke") or "smoke").lower()
        print(f"🧭 mode={mode}")

        if mode == "ltx2":
            result = generate_video_ltx2(job)
        else:
            result = generate_video_smoke(job)

        print(f"✅ Job {job_id} completed successfully")
        return result

    except Exception as e:
        print(f"❌ Job {job_id} failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "job_id": job_id
        }
# Initialize models on worker startup
print("🚀 Starting RunPod serverless worker (no startup model load)")
runpod.serverless.start({"handler": handler})
