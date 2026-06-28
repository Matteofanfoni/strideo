"""Video metadata schema and dataset lookup utilities.

Implements the metadata schema from the data collection protocol (§11.3)
using Pydantic models. Provides loading, saving, and lookup by video_id
so the pipeline can resolve runner/capture info automatically.

Reference: docs/technical/01_data_collection_protocol.md §11.3–11.4
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ── Schema models (§11.3) ──────────────────────────────────────────────


class FilesPaths(BaseModel):
    """Paths to raw and CFR video files."""

    raw_vfr: str
    cfr: Optional[str] = None


class RunnerInfo(BaseModel):
    """Runner identification and anthropometrics."""

    id: str
    height_cm: float
    stratum: str = Field(pattern=r"^[ABC]$")
    specialist_event: str


class PersonalBests(BaseModel):
    """Personal best times for key events."""

    model_config = {"populate_by_name": True, "extra": "allow"}

    pb_800m: Optional[str] = Field(None, alias="800m")
    pb_1500m: Optional[str] = Field(None, alias="1500m")


class FootwearInfo(BaseModel):
    """Footwear details for biomechanical context."""

    category: str
    model: str
    approximate_age_months: Optional[int] = None


class CaptureInfo(BaseModel):
    """Video capture conditions."""

    camera_distance_m: float
    stabilisation: str = "off"
    lens: str = "1x"
    resolution: str = "4K"
    fps: int = 60
    duration_s: Optional[float] = None
    pace_level: str


class QualityInfo(BaseModel):
    """Post-processing quality flags (populated after pipeline run)."""

    valid: Optional[bool] = None
    detection_rate: Optional[float] = None
    calibration_confidence: Optional[float] = None
    duration_flag: Optional[str] = None


class VideoMetadata(BaseModel):
    """Complete metadata for a single video, matching §11.3 schema."""

    video_id: str
    protocol_version: str = "1.5"
    files: FilesPaths
    runner: RunnerInfo
    footwear: FootwearInfo
    capture: CaptureInfo
    quality: QualityInfo = Field(default_factory=QualityInfo)


# ── Pipeline output (Technical Pipeline Part 7) ──────────────────────


class CalibrationOutput(BaseModel):
    """Spatial calibration results for pipeline output."""

    method: str = "direct"
    confidence: float
    pixels_per_cm: float
    segments_px: Dict[str, float]
    segments_proportions: Dict[str, float]
    proportion_warnings: List[str]
    leg_length_cm: float


class TemporalMetrics(BaseModel):
    """Temporal biomechanical metrics."""

    cadence_spm: float
    cadence_std: float
    gct_ms: float
    gct_std: float
    flight_time_ms: float
    duty_factor: float


class SpatialAbsoluteMetrics(BaseModel):
    """Spatial metrics in absolute units."""

    stride_length_m: float
    stride_length_std: float
    step_length_m: float
    oscillation_cm: float
    leg_length_cm: float


class SpatialNormalisedMetrics(BaseModel):
    """Spatial metrics normalised to leg length."""

    stride_leg_ratio: float
    oscillation_leg_ratio: float


class EfficiencyMetrics(BaseModel):
    """Running efficiency metrics."""

    running_economy_index: float
    flight_ratio: float


class DerivedMetrics(BaseModel):
    """Metrics derived from other metrics."""

    velocity_ms: float
    velocity_kmh: float
    pace_per_km: str


class MetricsOutput(BaseModel):
    """All biomechanical metrics, grouped by category."""

    temporal: TemporalMetrics
    spatial_absolute: SpatialAbsoluteMetrics
    spatial_normalised: SpatialNormalisedMetrics
    efficiency: EfficiencyMetrics
    derived: DerivedMetrics


class QualityOutput(BaseModel):
    """Quality indicators for the pipeline run."""

    detection_rate: float
    mean_hip_visibility: float
    n_contacts: int
    n_refined_contacts: int
    metric_warnings: List[str]
    overall_score: float


class InputInfo(BaseModel):
    """Record of what was fed into the pipeline."""

    video_path: str
    video_id: str
    runner_height_cm: float
    shoe_sole_cm: float
    fps: float
    duration_seconds: float


class PipelineResult(BaseModel):
    """
    Complete pipeline output for a single video pass.

    Matches the JSON format specified in Technical Pipeline Part 7.
    One file per pass, stored at ``data/results/{video_id}.json``.
    """

    version: str = "1.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    input: InputInfo
    calibration: CalibrationOutput
    metrics: MetricsOutput
    quality: QualityOutput


def load_result(path: str) -> PipelineResult:
    """Load a pipeline result JSON file.

    Args:
        path: Path to the result JSON file.

    Returns:
        Parsed PipelineResult object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return PipelineResult(**data)


def save_result(result: PipelineResult, path: str) -> None:
    """Save a PipelineResult to a JSON file.

    Args:
        result: The pipeline result to save.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


# ── I/O utilities ──────────────────────────────────────────────────────


def load_metadata(path: str) -> VideoMetadata:
    """
    Load a single video metadata JSON file.

    Args:
        path: Path to the metadata JSON file.

    Returns:
        Parsed VideoMetadata object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    return VideoMetadata(**data)


def save_metadata(metadata: VideoMetadata, path: str) -> None:
    """
    Save a VideoMetadata object to a JSON file.

    Args:
        metadata: The metadata to save.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )


# ── Dataset-level lookup ──────────────────────────────────────────────


class MetadataStore:
    """
    In-memory index of all video metadata in a dataset directory.

    Loads all .json files from a directory and indexes them by video_id
    for fast lookup during batch pipeline runs.

    Args:
        metadata_dir: Directory containing per-video metadata JSON files.
    """

    def __init__(self, metadata_dir: str) -> None:
        self._dir = Path(metadata_dir)
        self._index: Dict[str, VideoMetadata] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Metadata directory not found: {self._dir}")

        for json_file in sorted(self._dir.glob("*.json")):
            meta = load_metadata(str(json_file))
            self._index[meta.video_id] = meta

    def get(self, video_id: str) -> VideoMetadata:
        """
        Look up metadata by video_id.

        Args:
            video_id: The video identifier (e.g. 'A_AM01_threshold_1').

        Returns:
            VideoMetadata for the requested video.

        Raises:
            KeyError: If video_id is not found in the store.
        """
        if video_id not in self._index:
            available = ", ".join(sorted(self._index.keys()))
            raise KeyError(
                f"Video '{video_id}' not found. " f"Available: {available or '(none)'}"
            )
        return self._index[video_id]

    def list_ids(self) -> list[str]:
        """Return all video IDs in the store."""
        return sorted(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, video_id: str) -> bool:
        return video_id in self._index


# ── Runner database (§11.4) ──────────────────────────────────────────


class Runner(BaseModel):
    """Single runner entry matching §11.4 schema."""

    id: str
    name: str
    height_cm: float
    stratum: str = Field(pattern=r"^[ABC]$")
    sex: str = Field(pattern=r"^[MF]$")
    age: int
    specialist_event: str
    personal_bests: PersonalBests = PersonalBests()
    footwear: FootwearInfo
    consent_signed: bool = False
    consent_date: Optional[str] = None


class RunnerDatabaseSchema(BaseModel):
    """Top-level schema for the runner database JSON file."""

    runners: List[Runner]


class RunnerDatabase:
    """
    In-memory index of all runners in the study.

    Loads the runner database JSON (§11.4) and indexes by runner ID
    for lookup when creating per-video metadata.

    Args:
        path: Path to the runner database JSON file.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._index: Dict[str, Runner] = {}
        self._load(path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Runner database not found: {path}")

        data = json.loads(p.read_text(encoding="utf-8"))
        db = RunnerDatabaseSchema(**data)
        for runner in db.runners:
            self._index[runner.id] = runner

    def get(self, runner_id: str) -> Runner:
        """
        Look up a runner by ID.

        Args:
            runner_id: The runner identifier (e.g. 'AM01').

        Returns:
            Runner object.

        Raises:
            KeyError: If runner_id is not found.
        """
        if runner_id not in self._index:
            available = ", ".join(sorted(self._index.keys()))
            raise KeyError(
                f"Runner '{runner_id}' not found. "
                f"Available: {available or '(none)'}"
            )
        return self._index[runner_id]

    def to_runner_info(self, runner_id: str) -> RunnerInfo:
        """
        Convert a Runner entry to the RunnerInfo sub-model used in VideoMetadata.

        Args:
            runner_id: The runner identifier.

        Returns:
            RunnerInfo populated from the runner database.
        """
        runner = self.get(runner_id)
        return RunnerInfo(
            id=runner.id,
            height_cm=runner.height_cm,
            stratum=runner.stratum,
            specialist_event=runner.specialist_event,
        )

    def list_ids(self) -> list[str]:
        """Return all runner IDs."""
        return sorted(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, runner_id: str) -> bool:
        return runner_id in self._index
