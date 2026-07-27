# Training / evaluation image. Keeps CUDA + PyTorch + the LoRA stack.
# Build on a machine with a Docker daemon, then import with enroot on the cluster:
#   docker build -f containers/train.Dockerfile -t nebius-poc-train:dev .
#   enroot import -o nebius-poc-train.sqsh dockerd://nebius-poc-train:dev
#
# Pin the base tag to whatever discovery reports for the cluster driver.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nebius-poc

# Install the Python deps first so code edits do not bust the heavy layer.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir \
        "accelerate>=0.34" \
        "datasets>=2.19" \
        "numpy>=1.26" \
        "peft>=0.13" \
        "pyyaml>=6.0" \
        "transformers>=4.56" \
    && pip install --no-cache-dir -e . --no-deps

COPY configs ./configs
COPY scripts ./scripts

WORKDIR /workspace
CMD ["python", "-m", "nebius_poc.train", "--help"]
