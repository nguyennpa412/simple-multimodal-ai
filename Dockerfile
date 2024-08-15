FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /SMAI

# SYSTEM & PYTHON 3.10
RUN apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        software-properties-common build-essential apt-utils wget curl vim git ca-certificates kmod nvidia-driver-555 && \
    rm -rf /var/lib/apt/lists/* && \
    add-apt-repository --yes ppa:deadsnakes/ppa && \
    apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils python3.10-lib2to3 python3.10-gdbm python3.10-tk python3-pip python3-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 && \
    update-alternatives --config python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    apt-get autoclean

# VENV
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# INSTALL & COPY
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir wheel && \
    python3 -m pip install --no-cache-dir -r /SMAI/requirements.txt && \
    huggingface-cli download OpenGVLab/Mini-InternVL-Chat-2B-V1-5 && \
    huggingface-cli download microsoft/Florence-2-large
COPY . .

# RUN
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "/SMAI/app.py"]