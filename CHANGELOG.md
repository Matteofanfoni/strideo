# Changelog

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
