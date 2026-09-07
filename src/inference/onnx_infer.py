"""ONNX-backed inference for the shipped fast path.

``StrideoInference`` (``infer.py``) needs PyTorch. This module needs only
``numpy`` + ``onnxruntime`` (already a dependency, via RTMPose extraction),
so the app can run the shipped FiLM CNN without adding ``torch`` to
``requirements-app.txt`` -- the whole point of the ONNX-export choice
over a raw PyTorch dependency.

Mirrors ``StrideoInference``'s public API (``predict_window``/``predict``)
field-for-field, so app code migrates by changing only construction (an
``.onnx`` path instead of a ``.pt`` path or a live ``StrideoNet``), not call
sites. The exported graph (``export_to_onnx``'s ``_PredictModule`` wrapper,
which traces ``StrideoNet.predict`` rather than ``forward``) already
outputs denormalized physical units, so this module never needs
``BiomechanicsNormalizer``'s bounds -- there is exactly one place
(``film_net.py``) that knows the min/max ranges, and this module isn't it.

Averaging windows *after* each is already denormalized (rather than
averaging normalized values once, then denormalizing, as
``StrideoInference.predict`` does) is not an approximation of that
approach -- it is the identical computation. ``denormalize(x) = x * ranges
+ mins`` is per-metric affine with no clamping, and affine transforms
commute with averaging: ``mean(x_i) * range + min == mean(x_i * range +
min)``. Both wrappers must keep producing the same number for the same
checkpoint; if a future change to either normalizer breaks that identity,
it needs to change in both places.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import onnxruntime

# Stride has no output head -- mirrors infer.py's own floor
# and docstring exactly (see infer.py::_stride_post_hoc for the rationale).
_MIN_CADENCE_SPM = 60.0


def _stride_post_hoc(cadence_spm: float, velocity_ms: float) -> float:
    """stride_m = distance per stride = velocity (m/s) / strides-per-second.

    Mirrors the training-side post-hoc stride derivation exactly, including its
    factor-of-2 fix (a stride is two steps, as ``calculate_cadence`` counts
    them) -- see that module's docstring for the full rationale.
    """
    safe_cadence = max(cadence_spm, _MIN_CADENCE_SPM)
    return velocity_ms * 120.0 / safe_cadence


class OnnxStrideoInference:
    """Inference wrapper backed by ``onnxruntime`` instead of PyTorch.

    Accepts an ``.onnx`` file produced by
    ``src/inference/infer.py::export_to_onnx`` (equivalently,
    ``scripts/export_final_model_to_onnx.py``). Uses only the CPU execution
    provider -- the app's own deployment target (per
    ``requirements-app.txt``) has no GPU to route onnxruntime to.
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        window_size: int = 60,
        stride: int = 30,
    ):
        self.window_size = window_size
        self.stride = stride
        self.session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    def predict_window(self, poses: np.ndarray, pace: float) -> Dict:
        """Single 60-frame window -> physical-unit biomechanics for that
        window alone (already denormalized by the exported graph)."""
        p = poses[np.newaxis].astype(np.float32)
        v = np.array([[pace]], dtype=np.float32)
        biomechanics, confidence = self.session.run(
            ["biomechanics", "confidence"], {"poses": p, "pace": v}
        )
        return {
            "biomechanics": biomechanics[0],
            "confidence": float(confidence[0, 0]),
        }

    def predict(
        self,
        poses: np.ndarray,
        pace: float,
        confidence_weighted: bool = False,
        velocity_ms: Optional[float] = None,
    ) -> Dict:
        """Full sequence -> physical-unit biomechanics, averaged over
        windows. See ``StrideoInference.predict``'s own docstring for the
        ``confidence_weighted``/``velocity_ms`` argument semantics --
        unchanged here, including the default of ``False``.
        """
        T = len(poses)
        if T < self.window_size:
            raise ValueError(f"Sequence too short: {T} < {self.window_size}")

        windows = [
            poses[s : s + self.window_size]
            for s in range(0, T - self.window_size + 1, self.stride)
        ]
        preds = [self.predict_window(w, pace) for w in windows]

        confs = np.array([p["confidence"] for p in preds])
        weights = (
            confs / confs.sum()
            if confidence_weighted
            else np.ones(len(preds)) / len(preds)
        )

        avg_bio = sum(w * p["biomechanics"] for w, p in zip(weights, preds))
        avg_conf = float(np.mean(confs))

        cadence_spm = float(avg_bio[0])
        biomechanics = {
            "cadence_spm": cadence_spm,
            "gct_ms": float(avg_bio[1]),
            "oscillation_cm": float(avg_bio[2]),
        }
        if velocity_ms is not None:
            biomechanics["stride_length_m"] = _stride_post_hoc(cadence_spm, velocity_ms)

        return {
            "biomechanics": biomechanics,
            "confidence": avg_conf,
            "n_windows": len(windows),
        }
