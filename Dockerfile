# LTX2 RunPod Worker Dockerfile
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Fix torchvision mismatch with torch in base image (prevents "torchvision::nms does not exist")
RUN pip uninstall -y torchvision || true
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torchvision==0.17.0

# Clone LTX2 (monorepo)
RUN git clone https://github.com/Lightricks/LTX-2.git /tmp/ltx2

# Install LTX2 subpackages explicitly (this is the key fix)
RUN uv pip install --system /tmp/ltx2/packages/ltx-core
RUN uv pip install --system /tmp/ltx2/packages/ltx-pipelines

# Fix: base image torchvision is incompatible (operator torchvision::nms does not exist)
RUN python -c "import importlib.util; print('torchvision spec BEFORE:', importlib.util.find_spec('torchvision'))" || true
RUN uv pip uninstall -y torchvision || true
RUN python -c "import importlib.util; print('torchvision spec AFTER:', importlib.util.find_spec('torchvision'))" || true

# Force Gemma 3 capable transformers right before import
ENV TRANSFORMERS_NO_TORCHVISION=1

RUN uv pip install --system --upgrade --no-cache-dir \
    "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"

RUN python - <<'PY'
import traceback, transformers
print("transformers=", transformers.__version__)
try:
    from transformers import Gemma3ForConditionalGeneration
    print("Gemma3 class OK")
except Exception as e:
    print("Gemma3 import FAILED:", repr(e))
    traceback.print_exc()
    raise
PY

# ✅ SANITY CHECK (this makes the build fail if LTX2 isn’t importable)
RUN python -c "import ltx_core, ltx_pipelines; print('LTX2 import OK')"

# Create models directory
RUN mkdir -p /models

# Download LTX2 model files
RUN wget -O /models/ltx-2-19b-distilled-fp8.safetensors \
    https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors

RUN wget -O /models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors

RUN wget -O /models/ltx-2-19b-distilled-lora-384.safetensors \
    https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors

# Download Gemma text encoder (smaller version for faster startup)
#RUN git clone https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized /models/gemma-3-12b

# Set back to app directory
WORKDIR /app
ENV PYTHONPATH=/app
# Copy worker code
COPY runpod_worker.py .

# Set environment variables
ENV MODEL_PATH="/models"
ENV PYTHONPATH="/tmp/ltx2:$PYTHONPATH"

# Expose port (for debugging)
EXPOSE 8000

# Run the worker
CMD ["python", "runpod_worker.py"]
