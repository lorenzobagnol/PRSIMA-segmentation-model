FROM ubuntu:24.04
RUN apt-get update && apt-get install -y wget 

WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN /root/Miniconda3-latest-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN /root/miniconda3/bin/conda create --name main python==3.12
RUN echo 'conda activate main' >> ~/.bashrc
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

WORKDIR /workspace/
