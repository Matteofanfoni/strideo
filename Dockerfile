FROM python:3.11-slim

# System dependencies:
#   ffmpeg    — video decoding
#   wget      — model weight download at build time
#   libgles2  — OpenGL ES runtime required by MediaPipe (headless)
#   libegl1   — EGL platform library required by MediaPipe (headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg wget libgles2 libegl1 && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces requires the app to run as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
# Force MediaPipe to use CPU delegate — GPU delegate fails silently in headless Docker
ENV MEDIAPIPE_DISABLE_GPU=1

WORKDIR /home/user/app

# Install Python dependencies before copying the rest of the code so Docker
# can cache this layer and skip re-installation on code-only changes.
COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy application code
COPY --chown=user . .

# Download MediaPipe Pose Landmarker Heavy at build time (~30 MB).
# Baking it into the image avoids a download stall on every cold start.
RUN mkdir -p models && \
    wget -q -O models/pose_landmarker_heavy.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" && \
    ls -lh models/pose_landmarker_heavy.task

# Warm RTMPose-x ONNX — rtmlib auto-downloads the model (~300 MB) on first
# instantiation; baking it here puts it in the image layer so users never
# wait for a mid-request download.
RUN python -c "\
from rtmlib import Wholebody; \
Wholebody(mode='performance', to_openpose=False, backend='onnxruntime', device='cpu')"

EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
