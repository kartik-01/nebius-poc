# Multi-stage validator image.
# Build stage has the CUDA toolkit to compile gpu_smoke. Runtime stage keeps only
# the CUDA runtime, Python, fio, and inventory utilities — no compiler, no MPI,
# no ML frameworks.

ARG CUDA_IMAGE_TAG=12.4.1
ARG UBUNTU_TAG=ubuntu22.04

FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-${UBUNTU_TAG} AS build

WORKDIR /src
COPY validator/gpu_smoke.cu /src/gpu_smoke.cu
RUN nvcc -O2 -o /src/gpu_smoke /src/gpu_smoke.cu


FROM nvidia/cuda:${CUDA_IMAGE_TAG}-runtime-${UBUNTU_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        fio \
        jq \
        pciutils \
        numactl \
        ibverbs-utils \
        infiniband-diags \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Inventory helpers are best-effort. ibverbs packages may be absent on some
# base images; the validator treats missing IB tools as NOT_OBSERVABLE.

COPY --from=build /src/gpu_smoke /usr/local/bin/gpu_smoke
COPY validator/ /opt/validator/
COPY configs/validator.yaml /opt/validator/configs/validator.yaml

RUN python3 -m pip install --no-cache-dir pyyaml \
    && chmod +x /usr/local/bin/gpu_smoke

ENV PYTHONPATH=/opt

WORKDIR /opt
# No ENTRYPOINT — Pyxis/enroot pass the full command from the sbatch file, and an
# ENTRYPOINT would prepend and break that. `docker run` without args still works
# via CMD.
CMD ["python3", "-m", "validator.cluster_validate", \
     "--config", "/opt/validator/configs/validator.yaml", \
     "--out", "/results"]
