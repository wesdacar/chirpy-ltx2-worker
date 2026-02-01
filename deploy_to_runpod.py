"""
Automated RunPod Deployment Script for LTX2
Sets up serverless endpoint with proper configuration
"""

import requests
import json
import time
import os
from typing import Dict, Optional

class RunPodDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://rest.runpod.io/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_endpoint(
        self,
        name: str = "ltx2-chirpy-production",
        docker_image: str = "your-registry/ltx2-worker:latest",
        gpu_types: list = ["NVIDIA H100", "NVIDIA A100"],
        min_workers: int = 0,
        max_workers: int = 5,
        execution_timeout: int = 600000,  # 10 minutes
        idle_timeout: int = 5,
        env_vars: Dict = None
    ) -> Optional[str]:
        """
        Create a new RunPod serverless endpoint
        
        Returns:
            endpoint_id if successful, None if failed
        """
        
        if env_vars is None:
            env_vars = {
                "MODEL_PATH": "/models",
                "PYTHONPATH": "/tmp/ltx2:$PYTHONPATH"
            }
        
        payload = {
            "name": name,
            "computeType": "GPU",
            "gpuTypeIds": gpu_types,
            "workersMin": min_workers,
            "workersMax": max_workers,
            "executionTimeoutMs": execution_timeout,
            "idleTimeout": idle_timeout,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 2,  # Scale up when queue has 2+ jobs
            "flashboot": True,  # Faster cold starts
            "dockerArgs": {
                "imageName": docker_image,
                "environmentVariables": env_vars
            },
            "containerDiskInGb": 50,  # Large enough for models
            "allowedCudaVersions": ["12.1"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/endpoints",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 201:
                result = response.json()
                endpoint_id = result.get("id")
                print(f"✅ Endpoint created successfully: {endpoint_id}")
                return endpoint_id
            else:
                print(f"❌ Failed to create endpoint: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating endpoint: {str(e)}")
            return None
    
    def get_endpoint_status(self, endpoint_id: str) -> Dict:
        """Check endpoint status and worker health"""
        try:
            response = requests.get(
                f"{self.base_url}/endpoints/{endpoint_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get status: {response.text}"}
                
        except Exception as e:
            return {"error": f"Error getting status: {str(e)}"}
    
    def wait_for_endpoint_ready(self, endpoint_id: str, timeout: int = 600) -> bool:
        """Wait for endpoint to be ready for requests"""
        start_time = time.time()
        
        print(f"⏳ Waiting for endpoint {endpoint_id} to be ready...")
        
        while time.time() - start_time < timeout:
            status = self.get_endpoint_status(endpoint_id)
            
            if "error" in status:
                print(f"❌ Error checking status: {status['error']}")
                return False
            
            # Check if any workers are running
            workers = status.get("workers", [])
            ready_workers = [w for w in workers if w.get("status") == "READY"]
            
            if ready_workers:
                print(f"✅ Endpoint ready with {len(ready_workers)} worker(s)")
                return True
            
            print(f"⏳ Still waiting... ({len(workers)} workers initializing)")
            time.sleep(30)
        
        print(f"⏰ Timeout waiting for endpoint to be ready")
        return False
    
    def test_endpoint(self, endpoint_id: str) -> bool:
        """Send a test request to verify endpoint works"""
        test_payload = {
            "input": {
                "prompt": "A simple test video of a sunset",
                "duration": 3,
                "quality": "fast",
                "width": 720,
                "height": 480
            }
        }
        
        try:
            response = requests.post(
                f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
                headers=self.headers,
                json=test_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("output", {})
                
                if output.get("success"):
                    print(f"✅ Test successful! Generated video: {output.get('video_id')}")
                    return True
                else:
                    print(f"❌ Test failed: {output.get('error')}")
                    return False
            else:
                print(f"❌ Test request failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {str(e)}")
            return False

def main():
    """Main deployment workflow"""
    
    # Get API key from environment
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return
    
    # Get Docker image name
    docker_image = os.getenv("DOCKER_IMAGE", "your-registry/ltx2-worker:latest")
    
    deployer = RunPodDeployer(api_key)
    
    print("🚀 Starting LTX2 deployment to RunPod...")
    
    # Create endpoint
    endpoint_id = deployer.create_endpoint(
        name="ltx2-chirpy-production",
        docker_image=docker_image,
        gpu_types=["NVIDIA A100", "NVIDIA H100"],
        min_workers=0,
        max_workers=3,
        env_vars={
            "MODEL_PATH": "/models",
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
            "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME", "")
        }
    )
    
    if not endpoint_id:
        print("❌ Failed to create endpoint")
        return
    
    # Wait for endpoint to be ready
    if not deployer.wait_for_endpoint_ready(endpoint_id):
        print("❌ Endpoint failed to become ready")
        return
    
    # Test the endpoint
    if deployer.test_endpoint(endpoint_id):
        print(f"""
✅ LTX2 deployment successful!

Endpoint ID: {endpoint_id}
Endpoint URL: https://api.runpod.ai/v2/{endpoint_id}

Add these to your environment variables:
RUNPOD_ENDPOINT_ID={endpoint_id}
RUNPOD_API_KEY={api_key}

You can now use the ChirpyVideoGenerator class in your application!
        """)
    else:
        print("❌ Deployment completed but test failed")

if __name__ == "__main__":
    main()