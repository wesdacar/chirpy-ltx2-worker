# LTX2 RunPod Implementation for Chirpy.me

Ready-to-deploy LTX2 video generation system to replace VEO3.1.

## 🎯 Quick Start

### 1. Prerequisites
- RunPod account with API key
- Docker Hub account (or other registry)
- AWS S3 bucket for video storage (optional)

### 2. Build and Push Docker Image
```bash
# Build the image
docker build -t your-username/ltx2-worker:latest .

# Push to registry
docker push your-username/ltx2-worker:latest
```

### 3. Deploy to RunPod
```bash
# Set environment variables
export RUNPOD_API_KEY="your_runpod_api_key"
export DOCKER_IMAGE="your-username/ltx2-worker:latest"
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"
export S3_BUCKET_NAME="your-video-bucket"

# Deploy
python deploy_to_runpod.py
```

### 4. Integration in Chirpy.me
```python
from chirpy_ltx2_client import ChirpyVideoGenerator

# Drop-in replacement for VEO3.1
generator = ChirpyVideoGenerator()

result = generator.generate(
    prompt="A beautiful sunset over mountains",
    duration=5,
    quality="high"
)

if result["success"]:
    print(f"Video URL: {result['video_url']}")
else:
    print(f"Error: {result['error']}")
```

## 📁 File Structure

```
LTX2_RunPod_Implementation/
├── runpod_worker.py          # Main worker code
├── chirpy_ltx2_client.py     # Client library for Chirpy.me
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── deploy_to_runpod.py       # Automated deployment
└── README.md                # This file
```

## ⚙️ Configuration

### Environment Variables
- `RUNPOD_API_KEY`: Your RunPod API key
- `RUNPOD_ENDPOINT_ID`: Endpoint ID (set after deployment)
- `AWS_ACCESS_KEY_ID`: AWS credentials (optional)
- `AWS_SECRET_ACCESS_KEY`: AWS credentials (optional)
- `S3_BUCKET_NAME`: S3 bucket for video storage (optional)

### Video Generation Parameters
- **Prompt**: Text description (required)
- **Duration**: 1-10 seconds (default: 5)
- **Quality**: "fast" (8 steps) or "high" (40 steps)
- **Resolution**: HD (1280x720), FHD (1920x1080), etc.
- **FPS**: 24 or 30 frames per second

## 💰 Cost Estimates

### RunPod Pricing (A100 GPU)
- **Fast quality**: ~$0.05-0.08 per video (30-60s generation)
- **High quality**: ~$0.08-0.12 per video (2-4min generation)

### vs VEO3.1
- **Current VEO3.1**: ~$0.10-0.20 per video
- **LTX2 savings**: 50-60% cost reduction

## 🚀 Performance

### Generation Times (A100 GPU)
- **Fast quality**: 30-60 seconds
- **High quality**: 2-4 minutes
- **Cold start**: ~30 seconds (with flashboot)

### Auto-scaling
- Scales from 0 to 5 workers automatically
- Pay only when generating videos
- Sub-250ms cold start with pre-warmed workers

## 🛠️ Advanced Usage

### Async Generation
```python
client = LTX2Client(endpoint_id, api_key)

# Start generation
result = client.generate_video(
    prompt="Your prompt here",
    async_mode=True
)

job_id = result.job_id

# Check status later
final_result = client.wait_for_completion(job_id)
```

### Custom Resolutions
```python
from chirpy_ltx2_client import VideoResolution

result = client.generate_video(
    prompt="Your prompt",
    resolution=VideoResolution.FHD  # 1920x1080
)
```

## 🔧 Monitoring

### Check Endpoint Status
```python
from deploy_to_runpod import RunPodDeployer

deployer = RunPodDeployer(api_key)
status = deployer.get_endpoint_status(endpoint_id)
print(f"Workers: {len(status.get('workers', []))}")
```

### RunPod Console
Monitor your endpoint at: https://console.runpod.io/serverless

## 🐛 Troubleshooting

### Common Issues
1. **Model download timeout**: Increase container disk size
2. **GPU memory issues**: Use FP8 models or reduce batch size
3. **Cold starts**: Enable flashboot and increase min workers

### Logs
Check worker logs in RunPod console for detailed error information.

## 📈 Optimization Tips

1. **Use FP8 models** for lower memory usage
2. **Enable model caching** in RunPod for faster startups
3. **Tune worker counts** based on traffic patterns
4. **Use fast quality** for previews, high for final videos

## 🔒 Security

- API keys stored as environment variables
- S3 bucket access with minimal permissions
- No model files exposed publicly

## 📞 Support

For issues with this implementation:
1. Check RunPod worker logs
2. Verify environment variables
3. Test with simple prompts first

## 🚀 Migration from VEO3.1

1. Deploy LTX2 endpoint
2. Update environment variables
3. Replace import statement:
   ```python
   # OLD
   from veo_client import VideoGenerator
   
   # NEW  
   from chirpy_ltx2_client import ChirpyVideoGenerator
   ```
4. Test with subset of users
5. Gradual rollout to full user base