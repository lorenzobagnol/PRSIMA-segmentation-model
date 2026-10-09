# Annotation tool for Cloud Run (CPU only). Build context: this folder.
# The SAM 3 weights are gated on Hugging Face, so cloudbuild.yaml downloads them into ./weights
# before this build; they are baked into the image to avoid downloading 3.4GB at every cold start.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY weights/ /models/hf/
ENV HF_HOME=/models/hf HF_HUB_OFFLINE=1

COPY server.py sam3_engine.py unet_engine.py ./
COPY static/ ./static/

# Cloud Run sets PORT; the bucket is mounted at /data. UNET_WEIGHTS points at a checkpoint
# uploaded separately to the bucket (gs://<bucket>/models/<category>.pth) -- not baked into the
# image, since it changes every time the model is retrained and is small/fast to load.
ENV HOST=0.0.0.0 ANNOTATION_DATA=/data SAM3_DEVICE=cpu UNET_WEIGHTS=/data/models/distacco.pth
CMD ["python", "server.py"]
