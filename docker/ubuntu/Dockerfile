ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS gpu
ENV UV_ARGUMENTS="--extra all,cu126"
RUN echo "Using GPU image"

FROM ubuntu:22.04 AS cpu
ENV UV_ARGUMENTS="--extra all,cpu"
RUN echo "Using CPU-only image"

# --- Build ---
FROM ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV UV_ARGUMENTS=${UV_OVERRIDE:-$UV_ARGUMENTS}

ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      tzdata nano curl ffmpeg libsm6 libxext6 libnss3 \
      libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxrender1 \
      libasound2 libatk1.0-0 libgtk-3-0 libreoffice libjpeg-dev libpango-1.0-0 \
      libpangoft2-1.0-0 weasyprint && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid ${USER_GID} mmoreuser \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m mmoreuser

WORKDIR /app
RUN chown mmoreuser:mmoreuser /app
USER mmoreuser

# Set up Python virtual environment
RUN python3.11 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir uv

# Install dependencies (cached unless uv.lock changes)
COPY --chown=mmoreuser:mmoreuser pyproject.toml uv.lock /app/
RUN .venv/bin/uv sync --frozen --no-install-project ${UV_ARGUMENTS}

# Install mmore from local source code
COPY --chown=mmoreuser:mmoreuser . /app
RUN .venv/bin/uv pip install --no-cache --no-deps -e .

# ============================================
# 👇👇👇 这里自动下载全部 3 个必须模型
# ============================================
ARG PREDOWNLOAD_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG HF_ENDPOINT=https://hf-mirror.com

ENV HF_ENDPOINT=${HF_ENDPOINT}

# 下载向量模型 + mmore 核心模型（全部必须）
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading sentence transformer model'); \
    model = SentenceTransformer('${PREDOWNLOAD_MODEL}')"

# 👇 新增：自动下载 mmore 必须的 3 个核心模型
RUN mkdir -p /app/models && cd /app/models && \
    curl -L https://hf-mirror.com/MBZUAI/mmore/resolve/main/mmore_final_checkpoint.pth -O && \
    curl -L https://hf-mirror.com/MBZUAI/GeoLlama-7B-v1/resolve/main/config.json -O && \
    curl -L https://hf-mirror.com/MBZUAI/GeoLlama-7B-v1/resolve/main/generation_config.json -O

# --- Runtime ---
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

ENTRYPOINT ["/bin/bash"]
