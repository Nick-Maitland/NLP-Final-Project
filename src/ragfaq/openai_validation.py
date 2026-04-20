from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os

from .chunking import chunk_documents
from .config import PathConfig, ensure_runtime_directories, get_paths, get_runtime_availability
from .generation import ABSTENTION_TEXT, answer_question
from .ingest import load_documents
from .retrievers import build_lexical_index, lexical_index_exists, retrieve
from .schemas import BackendMode, LlmMode
from .utils import RagFaqError, dump_json

OPENAI_VALIDATION_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class OpenAIValidationCase:
    question: str
    expect_abstention: bool


OPENAI_VALIDATION_CASES = [
    OpenAIValidationCase(
        question="Why are embeddings usually more useful than one hot vectors for meaning?",
        expect_abstention=False,
    ),
    OpenAIValidationCase(
        question="How does self attention let a token use information from the rest of the sentence?",
        expect_abstention=False,
    ),
    OpenAIValidationCase(
        question="What is the capital city of France?",
        expect_abstention=True,
    ),
]

SKIPPED_NO_API_KEY = "skipped_no_api_key"
SKIPPED_RUN_LIVE_NOT_REQUESTED = "skipped_run_live_not_requested"
SKIPPED_OPENAI_SDK_UNAVAILABLE = "skipped_openai_sdk_unavailable"
PASSED = "passed"
FAILED = "failed"
NON_FAILING_STATUSES = {
    SKIPPED_NO_API_KEY,
    SKIPPED_RUN_LIVE_NOT_REQUESTED,
    SKIPPED_OPENAI_SDK_UNAVAILABLE,
    PASSED,
}


def _summary(
    *,
    status: str,
    run_live: bool,
    reason: str,
    case_results: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    case_results = case_results or []
    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_live": run_live,
        "model_name": OPENAI_VALIDATION_MODEL,
        "question_count": len(OPENAI_VALIDATION_CASES),
        "reason": reason,
        "cases": case_results,
    }


def _write_summary(paths: PathConfig, payload: dict[str, object]) -> None:
    dump_json(paths.openai_validation_summary_path, payload)


def ensure_openai_validation_summary(
    payload: dict[str, object],
    paths: PathConfig | None = None,
) -> PathConfig:
    paths = ensure_runtime_directories(paths or get_paths())
    _write_summary(paths, payload)
    return paths


def _ensure_lexical_index(paths: PathConfig) -> None:
    if lexical_index_exists(paths):
        return
    documents = load_documents(paths)
    if not documents:
        raise RagFaqError(
            "No knowledge-base files were found. Populate the root knowledge_base/ folder first."
        )
    build_lexical_index(chunk_documents(documents), paths)


def _case_result(
    *,
    case: OpenAIValidationCase,
    passed: bool,
    failure_reasons: list[str],
    answer_text: str = "",
    abstained: bool = False,
    citation_warnings: list[str] | None = None,
    retrieved_source_ids: list[str] | None = None,
    resolved_backend: str = "",
    resolved_llm: str = "",
) -> dict[str, object]:
    citation_warnings = citation_warnings or []
    retrieved_source_ids = retrieved_source_ids or []
    return {
        "question": case.question,
        "expected_abstention": case.expect_abstention,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "answer_text": answer_text,
        "abstained": abstained,
        "citation_warnings": citation_warnings,
        "retrieved_source_ids": retrieved_source_ids,
        "resolved_backend": resolved_backend,
        "resolved_llm": resolved_llm,
    }


def run_openai_validation(
    *,
    run_live: bool,
    paths: PathConfig | None = None,
) -> dict[str, object]:
    if not os.environ.get("OPENAI_API_KEY"):
        return _summary(
            status=SKIPPED_NO_API_KEY,
            run_live=run_live,
            reason=(
                "OPENAI_API_KEY is not set. Skipping GPT-4o-mini validation and writing a "
                "skip summary without making any OpenAI API call."
            ),
        )

    if not run_live:
        return _summary(
            status=SKIPPED_RUN_LIVE_NOT_REQUESTED,
            run_live=False,
            reason=(
                "Skipping live GPT-4o-mini validation because --run-live was not passed. "
                "No OpenAI API call was made."
            ),
        )

    availability = get_runtime_availability()
    if not availability.openai_sdk.available:
        return _summary(
            status=SKIPPED_OPENAI_SDK_UNAVAILABLE,
            run_live=True,
            reason=f"OpenAI SDK is unavailable: {availability.openai_sdk.reason}",
        )

    paths = ensure_runtime_directories(paths or get_paths())

    try:
        _ensure_lexical_index(paths)
        case_results: list[dict[str, object]] = []
        all_passed = True

        for case in OPENAI_VALIDATION_CASES:
            failure_reasons: list[str] = []
            retrieval = retrieve(
                question=case.question,
                requested_backend=BackendMode.TFIDF,
                paths=paths,
            )
            answer = answer_question(
                question=case.question,
                retrieved_chunks=retrieval.chunks,
                requested_llm=LlmMode.OPENAI,
                resolved_backend=retrieval.resolved_backend,
            )
            if answer.resolved_backend is not BackendMode.TFIDF:
                failure_reasons.append(
                    f"expected resolved backend tfidf but received {answer.resolved_backend.value}"
                )
            if answer.resolved_llm is not LlmMode.OPENAI:
                failure_reasons.append(
                    f"expected resolved llm openai but received {answer.resolved_llm.value}"
                )

            if case.expect_abstention:
                if not answer.abstained:
                    failure_reasons.append("expected exact abstention for out-of-scope question")
                if answer.answer_text != ABSTENTION_TEXT:
                    failure_reasons.append("expected exact abstention text for out-of-scope question")
            else:
                if answer.abstained:
                    failure_reasons.append("unexpected abstention for grounded question")
                if answer.citation_warnings:
                    failure_reasons.append("grounded question returned citation warnings")

            case_passed = not failure_reasons
            all_passed = all_passed and case_passed
            case_results.append(
                _case_result(
                    case=case,
                    passed=case_passed,
                    failure_reasons=failure_reasons,
                    answer_text=answer.answer_text,
                    abstained=answer.abstained,
                    citation_warnings=answer.citation_warnings,
                    retrieved_source_ids=answer.sources,
                    resolved_backend=answer.resolved_backend.value,
                    resolved_llm=answer.resolved_llm.value,
                )
            )
    except Exception as exc:
        return _summary(
            status=FAILED,
            run_live=True,
            reason=f"{type(exc).__name__}: {exc}",
        )

    status = PASSED if all_passed else FAILED
    reason = "" if all_passed else "One or more live GPT-4o-mini validation checks failed."
    return _summary(
        status=status,
        run_live=True,
        reason=reason,
        case_results=case_results,
    )

