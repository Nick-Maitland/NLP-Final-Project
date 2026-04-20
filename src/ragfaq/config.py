from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from .schemas import BackendMode, LlmMode

APP_NAME = "ragfaq"
COLLECTION_NAME = "ragfaq_chunks"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_TOP_K = 3
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 12
DEFAULT_CHUNK_SIZE = 120
DEFAULT_CHUNK_OVERLAP = 24
HYBRID_RRF_K = 60
HYBRID_MMR_LAMBDA = 0.75
EMBED_DEVICE = "cpu"
EMBED_BATCH_SIZE = 16
ABSTENTION_CONFIDENCE_THRESHOLD = 0.58
ABSTENTION_MIN_CONTEXT_OVERLAP = 0.15
ABSTENTION_MIN_SENTENCE_QUESTION_OVERLAP = 2
ABSTENTION_MIN_SUPPORT_RATIO = 1.0
ABSTENTION_TOP_SIGNAL_FLOOR = 0.05
ABSTENTION_TFIDF_SCORE_NORMALIZER = 6.0
ABSTENTION_TFIDF_GAP_NORMALIZER = 2.0
ABSTENTION_CHROMA_DISTANCE_NORMALIZER = 1.2
ABSTENTION_CHROMA_GAP_NORMALIZER = 0.4


@dataclass(frozen=True)
class PathConfig:
    root_dir: Path
    knowledge_base_dir: Path
    data_dir: Path
    index_dir: Path
    cache_dir: Path
    results_dir: Path
    traces_dir: Path
    lexical_index_path: Path
    chunk_cache_path: Path
    chroma_dir: Path
    latest_trace_path: Path
    demo_run_path: Path
    test_questions_path: Path
    scored_questions_path: Path
    evaluation_summary_path: Path
    evaluation_report_path: Path
    failure_report_path: Path
    readme_path: Path
    requirements_path: Path


@dataclass(frozen=True)
class DependencyStatus:
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class RuntimeAvailability:
    chromadb: DependencyStatus
    sentence_transformers: DependencyStatus
    openai_sdk: DependencyStatus
    openai_key_available: bool


def get_root_dir() -> Path:
    override = os.environ.get("RAGFAQ_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_paths() -> PathConfig:
    root_dir = get_root_dir()
    data_dir = root_dir / ".ragfaq"
    index_dir = data_dir / "indexes"
    cache_dir = data_dir / "cache"
    results_dir = root_dir / "results"
    traces_dir = results_dir / "traces"
    return PathConfig(
        root_dir=root_dir,
        knowledge_base_dir=root_dir / "knowledge_base",
        data_dir=data_dir,
        index_dir=index_dir,
        cache_dir=cache_dir,
        results_dir=results_dir,
        traces_dir=traces_dir,
        lexical_index_path=index_dir / "tfidf_index.json",
        chunk_cache_path=index_dir / "chunks.json",
        chroma_dir=root_dir / "chroma_db",
        latest_trace_path=traces_dir / "latest_retrieval_trace.json",
        demo_run_path=results_dir / "demo_run.md",
        test_questions_path=root_dir / "test_questions.csv",
        scored_questions_path=results_dir / "test_questions_scored.csv",
        evaluation_summary_path=results_dir / "evaluation_summary.json",
        evaluation_report_path=results_dir / "evaluation_report.md",
        failure_report_path=root_dir / "failure_case_report.md",
        readme_path=root_dir / "README.md",
        requirements_path=root_dir / "requirements.txt",
    )


def _probe_dependency(module_name: str) -> DependencyStatus:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - reason formatting is runtime dependent
        return DependencyStatus(False, f"{type(exc).__name__}: {exc}")
    return DependencyStatus(True, "")


def _probe_spec_only(module_name: str) -> DependencyStatus:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:  # pragma: no cover - runtime dependent
        return DependencyStatus(False, f"{type(exc).__name__}: {exc}")
    if spec is None:
        return DependencyStatus(False, "module not installed")
    return DependencyStatus(True, "load deferred until backend use")


def get_runtime_availability() -> RuntimeAvailability:
    chroma_status = _probe_dependency("chromadb")
    sentence_status = _probe_dependency("sentence_transformers")
    openai_status = _probe_dependency("openai")
    return RuntimeAvailability(
        chromadb=chroma_status,
        sentence_transformers=sentence_status,
        openai_sdk=openai_status,
        openai_key_available=bool(os.environ.get("OPENAI_API_KEY")),
    )


def ensure_runtime_directories(paths: PathConfig | None = None) -> PathConfig:
    paths = paths or get_paths()
    paths.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.chroma_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.traces_dir.mkdir(parents=True, exist_ok=True)
    return paths


def dense_runtime_available(availability: RuntimeAvailability | None = None) -> tuple[bool, str]:
    availability = availability or get_runtime_availability()
    if not availability.chromadb.available:
        return False, f"chromadb unavailable: {availability.chromadb.reason}"
    if not availability.sentence_transformers.available:
        return (
            False,
            "sentence-transformers unavailable: "
            f"{availability.sentence_transformers.reason}",
        )
    return True, ""


def resolve_query_backend(
    requested: BackendMode,
    lexical_index_ready: bool,
    dense_index_ready: bool,
) -> BackendMode:
    if requested is not BackendMode.AUTO:
        return requested
    if lexical_index_ready and dense_index_ready:
        return BackendMode.HYBRID
    if dense_index_ready:
        return BackendMode.CHROMA
    return BackendMode.TFIDF


def resolve_llm_mode(
    requested: LlmMode,
    availability: RuntimeAvailability | None = None,
) -> LlmMode:
    if requested is LlmMode.AUTO:
        availability = availability or get_runtime_availability()
        return LlmMode.OPENAI if availability.openai_key_available else LlmMode.OFFLINE
    return requested
