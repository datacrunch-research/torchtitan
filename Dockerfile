FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace/torchtitan

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install nightly PyTorch
RUN pip3 install --force-reinstall --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install  -e . 

# Use source code (not package) to ensure compatibility with nightly PyTorch
ENV NCCL_DEBUG=INFO
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL
ENV TORCHELASTIC_ERROR_FILE=/tmp/torchelastic_error.json

EXPOSE 29500

CMD ["/bin/bash"]
