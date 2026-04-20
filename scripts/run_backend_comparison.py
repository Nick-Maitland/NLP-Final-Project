from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.chunking import chunk_documents
from ragfaq.comparison import sync_backend_comparison_docs, write_backend_comparison_table
from ragfaq.config import (
    COLLECTION_NAME,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    ensure_runtime_directories,
    get_paths,
    get_runtime_availability,
)
from ragfaq.evaluation import compute_evaluation_results, write_scored_results_csv
from ragfaq.ingest import load_documents
from ragfaq.retrievers import maybe_build_indexes
from ragfaq.schemas import BackendMode, EvaluationRow, LlmMode
from ragfaq.utils import RagFaqError, dump_json


@dataclass(frozen=True)
class ComparisonConfig:
    label: str
    slug: str
    requested_backend: BackendMode
    requested_llm: LlmMode
    required: bool = False


COMPARISON_CONFIGS = [
    ComparisonConfig(
        label="tfidf + offline",
        slug="tfidf_offline",
        requested_backend=BackendMode.TFIDF,
        requested_llm=LlmMode.OFFLINE,
        required=True,
    ),
    ComparisonConfig(
        label="auto + offline",
        slug="auto_offline",
        requested_backend=BackendMode.AUTO,
        requested_llm=LlmMode.OFFLINE,
        required=True,
    ),
    ComparisonConfig(
        label="chroma + offline",
        slug="chroma_offline",
        requested_backend=BackendMode.CHROMA,
        requested_llm=LlmMode.OFFLINE,
    ),
    ComparisonConfig(
        label="hybrid + offline",
        slug="hybrid_offline",
        requested_backend=BackendMode.HYBRID,
        requested_llm=LlmMode.OFFLINE,
    ),
    ComparisonConfig(
        label="chroma + openai",
        slug="chroma_openai",
        requested_backend=BackendMode.CHROMA,
        requested_llm=LlmMode.OPENAI,
    ),
    ComparisonConfig(
        label="hybrid + openai",
        slug="hybrid_openai",
        requested_backend=BackendMode.HYBRID,
        requested_llm=LlmMode.OPENAI,
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backend comparison across retrieval and generation configurations.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Run only offline configurations and record OpenAI modes as skipped.",
    )
    parser.add_argument(
        "--include-openai",
        action="store_true",
        help="Allow OpenAI configurations when OPENAI_API_KEY and optional dependencies are available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve per question.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Hybrid candidate pool size before fusion and MMR.",
    )
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Chroma collection name to use when dense retrieval is available.",
    )
    return parser


def _load_docs_and_chunks():
    paths = ensure_runtime_directories(get_paths())
    documents = load_documents(paths)
    if not documents:
        raise RagFaqError(
            "No knowledge-base files were found. Populate the root knowledge_base/ folder first."
        )
    return paths, chunk_documents(documents)


def _comparison_scored_csv_path(paths, slug: str) -> Path:
    return paths.comparisons_dir / f"{slug}_scored.csv"


def _relative_to_root(paths, target_path: Path | None) -> str | None:
    if target_path is None:
        return None
    try:
        return str(target_path.relative_to(paths.root_dir))
    except ValueError:
        return str(target_path)


def _resolved_mode(results: list[EvaluationRow], field_name: str) -> str | None:
    values = sorted({getattr(result, field_name) for result in results if getattr(result, field_name)})
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return ",".join(values)


def _summary_row(
    config: ComparisonConfig,
    *,
    status: str,
    reason: str = "",
    results: list[EvaluationRow] | None = None,
    summary: dict[str, object] | None = None,
    scored_csv_path: Path | None = None,
    paths=None,
) -> dict[str, object]:
    row = {
        "label": config.label,
        "slug": config.slug,
        "required": config.required,
        "status": status,
        "requested_backend": config.requested_backend.value,
        "requested_llm": config.requested_llm.value,
        "resolved_backend": None,
        "resolved_llm": None,
        "questions": None,
        "answerable_questions": None,
        "unanswerable_questions": None,
        "retrieval_recall_at_3": None,
        "mrr_at_3": None,
        "faithfulness": None,
        "citation_validity": None,
        "abstention_accuracy": None,
        "false_abstention_rate": None,
        "average_latency_ms": None,
        "artifact_path": _relative_to_root(paths, scored_csv_path) if paths is not None else None,
        "reason": reason,
    }
    if not results or summary is None:
        return row
    row.update(
        {
            "resolved_backend": _resolved_mode(results, "resolved_backend"),
            "resolved_llm": _resolved_mode(results, "resolved_llm"),
            "questions": summary["question_count"],
            "answerable_questions": summary["answerable_count"],
            "unanswerable_questions": summary["unanswerable_count"],
            "retrieval_recall_at_3": summary["retrieval_recall_at_3_answerable"],
            "mrr_at_3": summary["mrr_at_3_answerable"],
            "faithfulness": summary["faithfulness_avg"],
            "citation_validity": summary["citation_valid_rate"],
            "abstention_accuracy": summary["abstention_accuracy_unanswerable"],
            "false_abstention_rate": summary["false_abstention_rate_answerable"],
            "average_latency_ms": summary["avg_latency_ms"],
        }
    )
    return row


def _run_successful_config(
    config: ComparisonConfig,
    *,
    paths,
    top_k: int,
    candidate_k: int,
    collection_name: str,
) -> dict[str, object]:
    results, summary, _ = compute_evaluation_results(
        requested_backend=config.requested_backend,
        requested_llm=config.requested_llm,
        paths=paths,
        top_k=top_k,
        candidate_k=candidate_k,
        collection_name=collection_name,
    )
    scored_csv_path = _comparison_scored_csv_path(paths, config.slug)
    write_scored_results_csv(results, scored_csv_path)
    return _summary_row(
        config,
        status="success",
        results=results,
        summary=summary,
        scored_csv_path=scored_csv_path,
        paths=paths,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    include_openai = args.include_openai and not args.offline_only
    paths, chunks = _load_docs_and_chunks()
    runtime = get_runtime_availability()

    required_failures = False
    configs: list[dict[str, object]] = []
    dense_ready = False
    dense_reason = "dense backend has not been prepared"

    for config in COMPARISON_CONFIGS:
        if config.slug == "tfidf_offline":
            try:
                maybe_build_indexes(
                    chunks,
                    requested_backend=BackendMode.TFIDF,
                    paths=paths,
                    collection_name=args.collection_name,
                )
                row = _run_successful_config(
                    config,
                    paths=paths,
                    top_k=args.top_k,
                    candidate_k=args.candidate_k,
                    collection_name=args.collection_name,
                )
            except Exception as exc:
                required_failures = True
                row = _summary_row(config, status="failed", reason=f"{type(exc).__name__}: {exc}")
            configs.append(row)
            continue

        if config.slug == "auto_offline":
            try:
                auto_build_summary = maybe_build_indexes(
                    chunks,
                    requested_backend=BackendMode.AUTO,
                    paths=paths,
                    collection_name=args.collection_name,
                )
                dense_ready = bool(auto_build_summary["dense_index"].get("built"))
                dense_reason = auto_build_summary["dense_index"].get(
                    "reason",
                    "dense retrieval unavailable",
                )
                row = _run_successful_config(
                    config,
                    paths=paths,
                    top_k=args.top_k,
                    candidate_k=args.candidate_k,
                    collection_name=args.collection_name,
                )
            except Exception as exc:
                required_failures = True
                dense_ready = False
                dense_reason = f"{type(exc).__name__}: {exc}"
                row = _summary_row(config, status="failed", reason=dense_reason)
            configs.append(row)
            continue

        if config.requested_llm is LlmMode.OPENAI:
            if args.offline_only:
                configs.append(
                    _summary_row(
                        config,
                        status="skipped",
                        reason="openai disabled by --offline-only",
                    )
                )
                continue
            if not include_openai:
                configs.append(_summary_row(config, status="skipped", reason="openai disabled"))
                continue
            if not runtime.openai_sdk.available:
                configs.append(
                    _summary_row(
                        config,
                        status="skipped",
                        reason=f"openai sdk unavailable: {runtime.openai_sdk.reason}",
                    )
                )
                continue
            if not runtime.openai_key_available:
                configs.append(
                    _summary_row(config, status="skipped", reason="OPENAI_API_KEY missing")
                )
                continue
            if not dense_ready:
                configs.append(
                    _summary_row(
                        config,
                        status="skipped",
                        reason=f"dense retrieval unavailable: {dense_reason}",
                    )
                )
                continue

        if config.requested_backend in {BackendMode.CHROMA, BackendMode.HYBRID} and not dense_ready:
            configs.append(
                _summary_row(
                    config,
                    status="skipped",
                    reason=f"dense retrieval unavailable: {dense_reason}",
                )
            )
            continue

        try:
            row = _run_successful_config(
                config,
                paths=paths,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                collection_name=args.collection_name,
            )
        except Exception as exc:
            row = _summary_row(config, status="failed", reason=f"{type(exc).__name__}: {exc}")
        configs.append(row)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "offline_only": args.offline_only,
        "include_openai": include_openai,
        "openai_key_present": runtime.openai_key_available,
        "openai_sdk_available": runtime.openai_sdk.available,
        "comparison_count": len(configs),
        "successful_configs": sum(1 for config in configs if config["status"] == "success"),
        "skipped_configs": sum(1 for config in configs if config["status"] == "skipped"),
        "failed_configs": sum(1 for config in configs if config["status"] == "failed"),
        "configs": configs,
    }

    dump_json(paths.backend_comparison_summary_path, summary_payload)
    write_backend_comparison_table(paths.backend_comparison_table_path, summary_payload)
    sync_backend_comparison_docs(paths, summary_payload=summary_payload)

    print(f"Backend comparison summary: {paths.backend_comparison_summary_path}")
    print(f"Backend comparison table: {paths.backend_comparison_table_path}")
    for config in configs:
        message = f"- {config['label']}: {config['status']}"
        if config.get("reason"):
            message += f" ({config['reason']})"
        elif config.get("artifact_path"):
            message += f" -> {config['artifact_path']}"
        print(message)

    if required_failures:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RagFaqError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
