FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget

WORKDIR /root/

# Download correct installer depending on architecture
# amd64  -> Miniconda (official)
# arm64  -> Miniforge (community, conda-forge)
RUN --mount=type=cache,target=/root/.cache \
    ARCH="$(uname -m)" && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Installing Miniconda (x86_64)..." && \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        echo "Installing Miniforge (aarch64)..." && \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniconda.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash miniconda.sh -b -p /root/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH="/root/conda/bin:${PATH}"
RUN conda init && conda clean -afy


# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install the right version of python and the required packages
WORKDIR /app
COPY . .

# Create the env
RUN conda create -y -n degradi python=3.13 
RUN conda run --live-stream -n degradi pip install --no-cache-dir -r requirements.txt