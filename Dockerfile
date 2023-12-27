FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.8 \
    python3-pip \
    python3.8-venv && \
    rm -rf /var/lib/apt/lists

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --pre torch==2.3.0.dev20231218+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121 && \
    python3 -m pip install --no-cache-dir \
    accelerate \
    transformers \
    peft 

RUN python3 -m pip install  --no-cache-dir diffusers==0.25.0
RUN python3 -m pip install --no-cache-dir git+https://github.com/pytorch-labs/ao@54bcd5a10d0abbe7b0c045052029257099f83fd9

CMD ["/bin/bash"]