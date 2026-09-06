# Changelog

## [Unreleased] — 2026-08-24

### Added

- **Fast results**: every clip is now analyzed by Strideo's trained neural
  network by default — reading pose data directly, about 6x faster than the
  full frame-by-frame analysis. Cadence and vertical oscillation come
  straight from the model; ground contact time is comparable in accuracy to
  the full classical pipeline, not more accurate. A "Run Full Analysis"
  button is available on every result to also get the full per-contact
  breakdown, added alongside the fast estimate rather than replacing it
- Redesigned visual identity across the whole app: new brand mark, typography,
  and page layouts for Home, Upload, Recording Guide, and Results
- Privacy Policy, Terms of Service, and Cookie Policy pages, linked from a
  shared footer on every page
- reCAPTCHA v2 spam protection on the contact form, verified server-side
  before a message is forwarded
- Homepage visuals: a pipeline-stage flow diagram and two real pipeline-output
  images (pose overlay, spatial-calibration example) under "How it works" and
  "How it's validated"

### Fixed

- HF Spaces secret configuration (contact form endpoint, reCAPTCHA keys) was
  not reaching the deployed app — the Docker SDK only injects Repository
  secrets as environment variables, unlike the native Streamlit SDK's
  automatic bridging into `st.secrets`

## [1.0.0] — 2026-06-29

### Added

- Classical biomechanics extraction pipeline (v1.20): cadence, ground contact
  time, stride length, and vertical oscillation from a single side-on clip
- Hybrid pose estimation: MediaPipe BlazePose (33 landmarks) + RTMPose-x
  (COCO-WholeBody) for high-reliability ground contact detection
- Interactive frame-by-frame ground-contact verifier (scrubber) with skeleton
  and IC/TO overlays; generates a downloadable annotated H.264 video on demand
- Runner profile form: name, age, sex, height, primary/secondary event and PB
- VDOT-based pace predictor — estimates Threshold / 1500 m / 800 m target paces
  from entered PBs (Jack Daniels equivalency system); displayed as filming
  targets alongside the measured pace
- PDF report export (ReportLab): runner info, per-clip metrics table with ± std,
  pipeline diagnostics, elite range comparison, cross-run summary
- Elite range comparison: custom HTML range bars for cadence, GCT, stride
  length, and vertical oscillation vs published world-class 800 m / 1500 m ranges
- Cross-run comparison table for 2–3 clips at different paces
- Recording guide page with full Protocol v1.6 checklist (camera position,
  settings, backdrop, common mistakes)
- Per-session isolated upload directories — concurrent visitors on the hosted
  instance cannot clobber each other's clips
- Live demo on Hugging Face Spaces (Docker, free CPU tier)

### Validated

- 13 / 15 pipeline targets met vs Kinovea reference measurements across a
  5-clip pre-validation dataset (GCT ±10 ms target, spatial calibration <3 %
  error)
- App ↔ CLI metric parity confirmed exact (0.0000 % deviation on reference clip)
- Runtime ~2–3 min per clip on 2 vCPU (acceptable for the hosted free tier)

### Notes

- **Vertical oscillation is not yet validated against ground truth.** The value
  is produced by the pipeline and displayed with an "unvalidated" flag; treat it
  as experimental until a future release confirms it against reference measurements.
- A pace-conditioned convolutional neural network (FiLM architecture) is
  scaffolded but untrained pending data collection. All current outputs use the
  classical extraction path above.
- Final evaluation will be leave-one-out cross-validation across 18 subjects;
  the pre-validation dataset (5 clips) is a subset used to validate the pipeline
  before full data collection begins.
