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

# Clone and install LTX2
RUN git clone https://github.com/Lightricks/LTX-2.git /tmp/ltx2
WORKDIR /tmp/ltx2
RUN uv sync --frozen
RUN uv pip install --system -e .

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
RUN git clone https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized /models/gemma-3-12b

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
