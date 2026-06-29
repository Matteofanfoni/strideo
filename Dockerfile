FROM python:3.11-slim

# System dependencies:
#   ffmpeg        — video decoding
#   wget          — model weight download at build time
#   libgles2      — OpenGL ES runtime required by MediaPipe
#   libegl1       — EGL platform library required by MediaPipe
#   xvfb          — virtual framebuffer so MediaPipe can initialise its GL
#                   context in the headless container (no real display)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg wget libgles2 libegl1 xvfb xauth && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces requires the app to run as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH

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
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Warm RTMPose-x ONNX — rtmlib auto-downloads the model (~300 MB) on first
# instantiation; baking it here puts it in the image layer so users never
# wait for a mid-request download.
RUN python -c "\
from rtmlib import Wholebody; \
Wholebody(mode='performance', to_openpose=False, backend='onnxruntime', device='cpu')"

EXPOSE 7860

# xvfb-run provides a virtual framebuffer so MediaPipe's EGL context
# initialises correctly in the headless container.
CMD ["xvfb-run", "-a", "--server-args=-screen 0 1024x768x24", \
     "streamlit", "run", "app/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
