# Base image from NVIDIA PyTorch
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Set working directory
WORKDIR /workspace/torchtitan

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy repository files
COPY . .

# Install nightly PyTorch
RUN pip3 install --force-reinstall --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . 

# Use source code (not package) to ensure compatibility with nightly PyTorch
# Set Python path and environment variables for distributed training
ENV PYTHONPATH=/workspace/torchtitan:$PYTHONPATH
ENV NCCL_DEBUG=INFO
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL
ENV TORCHELASTIC_ERROR_FILE=/tmp/torchelastic_error.json

# Expose common ports for distributed training
EXPOSE 29500

# Default command
CMD ["/bin/bash"]
