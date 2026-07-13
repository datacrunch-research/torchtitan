# Llama 3 70B training image for cluster runs (no fp8).
# Base: latest NGC PyTorch container as of 2026-07.
FROM nvcr.io/nvidia/pytorch:26.06-py3

WORKDIR /workspace/torchtitan

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install Python dependencies from requirements.txt (symlink into .ci/docker/).
RUN pip3 install --no-cache-dir -r requirements.txt

# Install pinned nightly PyTorch stack. The NGC base torch can lag behind the
# distributed features torchtitan needs (e.g. reduce_scatter_tensor_coalesced
# used by TP), so we force-reinstall the current nightly.
RUN pip3 install --force-reinstall --pre torch torchvision torchao \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Install torchtitan itself.
RUN pip3 install .

# Bake the Llama-3.1-70B tokenizer into the image. Cluster nodes have no shared
# storage, so assets must ship inside the image. Pass the HF token at build time:
#   --secret id=hf_token,src=$HOME/.cache/huggingface/token
# Writes to ./assets/hf/Llama-3.1-70B, matching the llama3_70b config's
# hf_assets_path.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) \
    python scripts/download_hf_assets.py \
    --repo_id meta-llama/Llama-3.1-70B --assets tokenizer

ENV NCCL_DEBUG=WARN
ENV TORCHELASTIC_ERROR_FILE=/tmp/torchelastic_error.json

EXPOSE 29500

# Default entrypoint runs the 70B training script.
# Works for both single-node (docker run) and multi-node (torchrun PET_* env).
ENTRYPOINT ["/workspace/torchtitan/train_llama70b.sh"]
