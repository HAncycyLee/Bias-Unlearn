FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl ca-certificates unzip openssh-client \
    build-essential libaio-dev \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    python -m pip install -U pip wheel setuptools
