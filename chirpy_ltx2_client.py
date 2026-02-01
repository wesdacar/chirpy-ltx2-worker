"""
Chirpy.me LTX2 Client
Drop-in replacement for VEO3.1 API calls
"""

import requests
import time
import os
import logging
from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VideoQuality(Enum):
    FAST = "fast"      # 8 inference steps, ~30-60s generation
    HIGH = "high"      # 40 inference steps, ~2-4min generation

class VideoResolution(Enum):
    HD = (1280, 720)
    FHD = (1920, 1080)
    SQUARE = (1024, 1024)
    VERTICAL = (720, 1280)

@dataclass
class GenerationResult:
    """Result of video generation request"""
    success: bool
    video_url: Optional[str] = None
    video_id: Optional[str] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None
    job_id: Optional[str] = None

class LTX2Client:
    """Client for LTX2 video generation via RunPod"""
    
    def __init__(
        self, 
        endpoint_id: str,
        api_key: str,
        base_url: str = "https://api.runpod.ai/v2",
        timeout: int = 300,
        max_retries: int = 3
    ):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        quality: VideoQuality = VideoQuality.HIGH,
        resolution: VideoResolution = VideoResolution.HD,
        fps: int = 24,
        async_mode: bool = False
    ) -> GenerationResult:
        """
        Generate video using LTX2
        
        Args:
            prompt: Text description of the video
            duration: Video duration in seconds (1-10)
            quality: Generation quality (fast or high)
            resolution: Video resolution
            fps: Frames per second (24 or 30)
            async_mode: If True, returns immediately with job_id
        
        Returns:
            GenerationResult with video_url or error
        """
        
        width, height = resolution.value
        
        payload = {
            "input": {
                "prompt": prompt,
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "quality": quality.value
            }
        }
        
        # Choose endpoint based on sync/async mode
        endpoint = "runsync" if not async_mode else "run"
        url = f"{self.base_url}/{self.endpoint_id}/{endpoint}"
        
        try:
            logger.info(f"🎬 Starting video generation: {prompt[:50]}...")
            
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result_data = response.json()
            
            if async_mode:
                # Return job ID for async tracking
                job_id = result_data.get("id")
                return GenerationResult(
                    success=True,
                    job_id=job_id
                )
            else:
                # Synchronous result
                output = result_data.get("output", {})
                
                if output.get("success"):
                    logger.info("✅ Video generation completed successfully")
                    return GenerationResult(
                        success=True,
                        video_url=output.get("video_url"),
                        video_id=output.get("video_id"),
                        generation_time=output.get("generation_time")
                    )
                else:
                    error = output.get("error", "Unknown generation error")
                    logger.error(f"❌ Generation failed: {error}")
                    return GenerationResult(
                        success=False,
                        error=error
                    )
                    
        except requests.exceptions.Timeout:
            logger.error("⏰ Request timed out")
            return GenerationResult(
                success=False,
                error="Request timed out"
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Request failed: {str(e)}")
            return GenerationResult(
                success=False,
                error=f"Request failed: {str(e)}"
            )
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return GenerationResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def check_job_status(self, job_id: str) -> GenerationResult:
        """Check status of async job"""
        url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status")
            
            if status == "COMPLETED":
                output = data.get("output", {})
                return GenerationResult(
                    success=output.get("success", False),
                    video_url=output.get("video_url"),
                    video_id=output.get("video_id"),
                    generation_time=output.get("generation_time"),
                    error=output.get("error"),
                    job_id=job_id
                )
            elif status == "FAILED":
                return GenerationResult(
                    success=False,
                    error=data.get("error", "Job failed"),
                    job_id=job_id
                )
            else:
                # Still running
                return GenerationResult(
                    success=False,
                    error=f"Job still {status.lower()}",
                    job_id=job_id
                )
                
        except Exception as e:
            return GenerationResult(
                success=False,
                error=f"Status check failed: {str(e)}",
                job_id=job_id
            )
    
    def wait_for_completion(
        self, 
        job_id: str, 
        poll_interval: int = 10,
        max_wait_time: int = 600
    ) -> GenerationResult:
        """Wait for async job to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            result = self.check_job_status(job_id)
            
            if result.success or (result.error and "still" not in result.error):
                return result
            
            logger.info(f"⏳ Waiting for job {job_id}...")
            time.sleep(poll_interval)
        
        return GenerationResult(
            success=False,
            error="Job timed out",
            job_id=job_id
        )

# Convenience wrapper functions (drop-in replacements)
class ChirpyVideoGenerator:
    """Drop-in replacement for existing VEO3.1 integration"""
    
    def __init__(self):
        # Load from environment variables
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        api_key = os.getenv("RUNPOD_API_KEY")
        
        if not endpoint_id or not api_key:
            raise ValueError("RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY must be set")
        
        self.client = LTX2Client(endpoint_id, api_key)
    
    def generate(
        self, 
        prompt: str, 
        duration: int = 5,
        quality: str = "high"
    ) -> Dict:
        """
        Drop-in replacement for VEO3.1 generate() function
        
        Returns:
            Dict with 'success', 'video_url', and 'error' keys
        """
        quality_enum = VideoQuality.HIGH if quality == "high" else VideoQuality.FAST
        
        result = self.client.generate_video(
            prompt=prompt,
            duration=duration,
            quality=quality_enum
        )
        
        return {
            "success": result.success,
            "video_url": result.video_url,
            "error": result.error,
            "generation_time": result.generation_time,
            "video_id": result.video_id
        }

# Example usage
if __name__ == "__main__":
    # Test the client
    generator = ChirpyVideoGenerator()
    
    result = generator.generate(
        prompt="A serene lake at sunset with gentle waves",
        duration=5,
        quality="high"
    )
    
    if result["success"]:
        print(f"✅ Video generated: {result['video_url']}")
    else:
        print(f"❌ Generation failed: {result['error']}")