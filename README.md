---
title: Strideo
emoji: 🏃
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Strideo

Running biomechanics analysis from a smartphone video.
Point your phone at a runner, upload the clip, and get four core metrics in seconds.

Live demo → Hugging Face Spaces (coming soon)

---

## What it measures

| Metric | Unit | Typical elite 800m range |
|---|---|---|
| Cadence | steps/min | 190 – 210 |
| Ground Contact Time | ms | 115 – 130 |
| Stride Length | m | 3.8 – 4.4 |
| Vertical Oscillation | m | 0.06 – 0.10 |

Targets middle-distance runners (800 m / 1500 m). Results include a comparison
against published elite ranges and an exportable PDF report.

---

## How it works

1. **Pose estimation** — [MediaPipe BlazePose](https://google.github.io/mediapipe/solutions/pose.html)
   extracts 33 body landmarks per frame; RTMPose-x adds high-precision foot
   landmarks for contact detection.
2. **Ground contact detection** — ankle landmark vertical velocity is used to
   identify initial contact and toe-off events frame-by-frame.
3. **Metric extraction** — cadence, GCT, stride length, and vertical oscillation
   are derived from the detected contact sequence using classical biomechanical
   formulae.
4. **Elite comparison** — each metric is placed against published ranges for
   elite 800 m / 1500 m competitors.

### Validation

The extraction pipeline has been validated against Kinovea reference measurements
across a pre-validation dataset of 5 clips. Current pass rate: **13/15 pipeline
targets met** (v1.20). A pace-conditioned model for metric refinement is in
training — the current app uses the classical extraction path.

---

## Getting started

### Requirements

- Python 3.11
- `ffmpeg` on your PATH (video decoding)
- MediaPipe pose model weights (see below)

### Install

```bash
git clone https://github.com/Matteofanfoni/strideo.git
cd strideo
pip install -r requirements-app.txt
```

### Model weights

MediaPipe requires a `.task` model file that is not included in the repo
(too large for git). Download the **Pose Landmarker Heavy** model from the
[MediaPipe Models page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
and place it at `models/pose_landmarker_heavy.task`.

RTMPose-x weights are downloaded automatically on first run.

### Run

```bash
streamlit run app/app.py
```

Then open [http://localhost:8501](http://localhost:8501), go to **Upload**, and
drop in a 4K 60 fps clip. Manual pace entry is required for the current MVP.

---

## Recording guide

For best results:

- **Camera:** iPhone 4K 60 fps, landscape orientation
- **Position:** tripod or stable mount, side-on to the runner, ~15 m away
- **Clothing:** fitted (tight-fitting top and shorts) for clean landmark detection
- **Pace:** steady-state effort; note the target pace before recording

The app's **Recording Guide** page has the full checklist.

---

## Directory structure

```
strideo/
├── app/
│   ├── app.py                  # Streamlit entry point
│   ├── pages/
│   │   ├── upload.py           # Video upload and pipeline trigger
│   │   ├── results.py          # Metric display, PDF export
│   │   └── recording_guide.py  # Recording protocol
│   └── ui/                     # Shared UI components and styles
├── src/
│   ├── preprocessing/          # Pose estimation, contact detection, metric extraction
│   └── utils/                  # Shared utilities (metrics, pace, quality)
├── requirements-app.txt        # App dependencies (CPU, no GPU required)
├── packages.txt                # System packages for Hugging Face Spaces
└── run_app.sh                  # Convenience launch script
```

---

## Privacy

Video files are processed locally (or on the Hugging Face Space's server) and
are not stored or transmitted elsewhere. No account or login is required.

**Not medical advice.** Strideo is a research and coaching tool.
Results should be interpreted alongside qualified coaching or sports science support.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.
