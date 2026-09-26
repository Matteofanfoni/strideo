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

Running biomechanics from a smartphone video. Point a phone at a runner, upload
the clip, and get four gait metrics back, with no lab hardware and no wearable.

**[Live demo → strideo.org](https://strideo.org)** · hosted on [Hugging Face Spaces](https://huggingface.co/spaces/fanfonim/strideo)

![Strideo home page](docs/assets/screenshots/screenshot_home.png)

---

## What it measures

Cadence, ground contact time, stride length and vertical oscillation, plus duty
factor. The app compares each result against elite reference bands drawn from
the published literature, built at the speed you actually ran, and exports a PDF
report.

The bands are sex specific, and they follow the measured speed rather than a
nominal pace because every one of these metrics moves with speed. At 26 km/h,
roughly 800 m race pace:

| Metric | Unit | Elite male | Elite female |
|---|---|---|---|
| Cadence | steps/min | 182.6 – 218.6 | 198.7 – 234.7 |
| Ground contact time | ms | 133.8 – 181.8 | 118.0 – 166.0 |
| Stride length | m | 4.00 – 4.64 | 3.57 – 4.21 |
| Vertical oscillation | cm | 3.96 – 9.72 \* | 3.14 – 8.58 \* |
| Duty factor | | 0.243 – 0.283 | 0.243 – 0.283 |

\* Extrapolated rather than measured. No whole-cycle vertical oscillation
measurement exists above 22 km/h in any published population, so the 800 m and
1500 m oscillation bands are modelled, and the app labels them as such.

Each band is two standard deviations of the spread between elite athletes, so
the bands are wide on purpose. They are modelled estimates drawn from 16 primary
sources, not Strideo measurements, and they are for orientation, not targets or
a diagnosis. The 800 m tier is the weakest evidence behind them: none of the 16
sources measured a cohort of elite men at 800 m race pace, and two of them are
not peer-reviewed. Strideo targets middle-distance runners because
800 m and 1500 m form changes with effort: cadence, contact time and stride
length all shift as a runner moves from threshold toward race pace, and most
consumer tools average that shift away.

---

## How it works

1. **Constant-rate conversion.** Variable-frame-rate phone footage is rewritten
   to a constant 60 fps, so every timing measurement shares one clock.
2. **Pose estimation.** [MediaPipe BlazePose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
   tracks 33 body landmarks per frame and calibrates real-world scale from the
   runner's own height.
3. **StrideoNet.** A FiLM pace-conditioned 2D CNN (~277K parameters) reads those
   landmarks directly and predicts cadence, ground contact time and vertical
   oscillation, without running a second pose model and without detecting
   individual footstrikes. Stride length follows from the predicted cadence and
   the measured speed. It produces the
   app's default result, and runs roughly 6x faster than the path below because
   it skips the second pose model.
4. **Deterministic kinematics engine** (opt-in). RTMPose-x adds high-precision
   hip and ankle landmarks, and a rules-based engine detects initial contact and
   toe-off per footstrike from leg compression, the vertical hip-to-ankle
   distance, which cancels the camera's own movement because that shifts hip
   and ankle together. It also classifies
   forefoot against heel strike. Triggered per result with **Run in-depth analysis**,
   and reported alongside the fast estimate rather than replacing it.

---

## How it's validated

Strideo was checked against a purpose-collected dataset of 45 clips, 5 runners
and one session, in a balanced 5 x 3 x 3 design of runner by pace by repeat, at
genuine race paces (800 m, 1500 m and threshold) rather than treadmill or
jogging footage. Every clip was hand-annotated in [Kinovea](https://www.kinovea.org/)
for initial contact and toe-off on each foot. The system never sees that
annotation. Camera geometry was fixed and surveyed: 12.5 m from the running
line, with a per-session pixel scale.

Ground contact time is the one metric with a pre-registered target, ≤ 10 ms mean
absolute error:

| Estimator | GCT mean absolute error | How it was measured |
|---|---|---|
| StrideoNet (default) | 7.57 ms | Leave-one-runner-out cross-validation. No fold's model ever saw its test runner. |
| Deterministic engine (opt-in) | 3.99 ± 3.60 ms | The same 45 clips, against the same annotation. |

Both clear the target, though the two figures rest on different footing:

- Every constant producing the engine's lead was fitted on this same cohort.
  Remove its contact filter alone and the two methods come back to a tie.
- The engine's figure is flattered by roughly 0.19 ms. 18 of the 45 clips are
  scored on fewer contacts than they contain, and the contacts that go missing
  are the hard ones.
- StrideoNet's figure is held out runner by runner. The engine's is not.

**Cadence** comes out clean: a scored-set residual of +0.1%, with all 45 clips
inside ±2%. That is the set matched to the annotation. The raw average the
engine reports without any annotation is +0.2%, with every clip inside ±5%.
**Stride length** shows no significant bias across the cohort (about 0.5% low,
not significant with 5 runners), but single clips can be off by up to about 6%,
so it is not a per-clip accuracy claim. **Vertical oscillation** is the
least settled of the four. Its displayed value carries a correction fitted to
the annotation, and much of the error left over is the annotation's own
cycle-to-cycle spread rather than the estimator's, so no tolerance is claimed
for it yet.

**Scope.** This is a feasibility demonstration on 5 runners, at one venue, one
camera geometry and one frame rate, not a full external validation. It does not
support an accuracy claim for a different runner, venue, geometry or frame rate,
and nothing here is intended for clinical or diagnostic use. More runners is the
clearest lever on all of it, which is why a second collection round is next.

![Strideo results page](docs/assets/screenshots/screenshot_results.png)

---

## Getting started

### Requirements

- Python 3.11
- `ffmpeg` on your PATH, for video decoding
- MediaPipe pose model weights, see below

### Install

```bash
git clone https://github.com/Matteofanfoni/strideo.git
cd strideo
pip install -r requirements-app.txt
```

Dependencies are CPU-only; no GPU required.

### Model weights

The trained StrideoNet model ships with the repo at `models/strideo_final.onnx`,
so the default analysis path works on a fresh clone.

MediaPipe needs a `.task` file that is too large to commit. Download the
**Pose Landmarker Heavy** model from the
[MediaPipe Models page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
and place it at `models/pose_landmarker_heavy.task`. RTMPose-x weights, used
only by the opt-in full analysis, are fetched automatically on first run.

### Run

```bash
streamlit run app/app.py
```

Then open [http://localhost:8501](http://localhost:8501), go to **Upload**, and
drop in a 4K 60 fps clip. Pace is entered by hand in the current version.

---

## Recording guide

For best results:

- **Camera:** iPhone 4K 60 fps, landscape orientation
- **Position:** tripod or stable mount, side-on to the runner, about 13 m from
  the running line, at roughly hip height
- **Framing:** keep the whole body in frame across the capture zone
- **Clothing:** fitted top and shorts, so the landmarks track cleanly
- **Pace:** steady-state effort, and note the target pace before recording

The app's **Recording Guide** page has the full checklist and a plan view of the
setup.

---

## Repository layout

```
strideo/
├── app/
│   ├── app.py                  # Streamlit entry point and Home page
│   ├── pipeline_runner.py      # Drives a clip through the pipeline
│   ├── preload.py              # Warms the models at startup
│   ├── pages/
│   │   ├── upload.py           # Clip upload, runner and pace entry
│   │   ├── results.py          # Metrics, elite comparison, PDF export
│   │   ├── recording_guide.py  # Recording protocol
│   │   └── ...                 # Privacy, terms and cookie policy pages
│   ├── ui/                     # Shared components, styles, header and footer
│   ├── utils/                  # Elite reference ranges
│   └── assets/                 # Logo, favicon, legal text, figures
├── src/
│   ├── preprocessing/          # Frame-rate conversion, pose estimation, contact detection
│   ├── inference/              # StrideoNet ONNX runtime path
│   └── utils/                  # Metrics, pace and quality helpers
├── models/
│   └── strideo_final.onnx      # Trained StrideoNet, ~1.1 MB
├── Dockerfile                  # Image used by the Hugging Face Space
├── requirements-app.txt        # App dependencies, CPU only
├── packages.txt                # System packages for Hugging Face Spaces
└── run_app.sh                  # Convenience launch script
```

---

## Privacy

Video is processed locally, or on the Hugging Face Space's server, and is not
stored or transmitted elsewhere. No account or login is required.

Full details: [Privacy Policy](https://fanfonim-strideo.hf.space/privacy_policy) ·
[Terms of Service](https://fanfonim-strideo.hf.space/terms_of_service) ·
[Cookie Policy](https://fanfonim-strideo.hf.space/cookie_policy)

**Not medical advice.** Strideo is a research and coaching tool. Results should
be interpreted alongside qualified coaching or sports science support.

---

## License

[MIT](LICENSE), free to use, modify and distribute.
