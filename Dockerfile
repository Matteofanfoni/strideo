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
# Force MediaPipe (BlazePose, the fast path) to use its CPU delegate — the
# GPU delegate fails silently in headless Docker. This is unrelated to
# RTMPose's own CUDA use below: RTMPose goes through onnxruntime, not
# MediaPipe's delegate, and auto-detects CUDA at runtime instead
# (src.preprocessing.rtmpose_extractor.detect_device()). No base-image or
# driver setup needed here for that: the same image runs unmodified on a
# CPU-tier or GPU-tier HF Space — GPU tiers expose the NVIDIA driver via HF's
# container runtime, and requirements-app.txt's
# onnxruntime-gpu[cuda,cudnn]==1.26.0 brings its own CUDA/cuDNN user-space
# libs via pip (Linux-only bundling; that's why the Windows dev env instead
# uses conda-forge cudnn, see environment_rtmpose.yml).
ENV MEDIAPIPE_DISABLE_GPU=1

WORKDIR /home/user/app

# Install Python dependencies before copying the rest of the code so Docker
# can cache this layer and skip re-installation on code-only changes.
COPY --chown=user requirements-app.txt .
# rtmlib's own METADATA unconditionally requires plain `onnxruntime` (any
# version), with no awareness that `onnxruntime-gpu` already satisfies the
# same import. Both packages install into the same on-disk `onnxruntime/`
# directory, so a single `pip install -r requirements-app.txt` leaves
# whichever wheel's files were physically written last as the one that
# actually imports — empirically confirmed to be plain `onnxruntime`
# (latest, CPU-only), silently discarding CUDAExecutionProvider regardless
# of `onnxruntime-gpu`'s pin. The `uninstall` + `--force-reinstall
# --no-deps` below undoes that clobber deterministically, matching the
# clean single-package state already validated locally
# (environment_rtmpose.yml's conda env only ever shows onnxruntime-gpu).
# `pip uninstall` on a package that (in some future dependency change)
# turns out not to be installed exits 0, not an error, so this stays safe
# either way.
RUN pip install --no-cache-dir -r requirements-app.txt && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir --force-reinstall --no-deps "onnxruntime-gpu[cuda,cudnn]==1.26.0"

# Copy application code
COPY --chown=user . .
RUN chmod +x docker-entrypoint.sh

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

CMD ["bash", "docker-entrypoint.sh"]
