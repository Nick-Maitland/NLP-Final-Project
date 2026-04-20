from __future__ import annotations

from pathlib import Path

from .config import PathConfig, get_paths
from .utils import RagFaqError, load_json

BACKEND_COMPARISON_START = "<!-- backend-comparison:start -->"
BACKEND_COMPARISON_END = "<!-- backend-comparison:end -->"
COMPARISON_SUMMARY_RELATIVE_PATH = "results/comparisons/backend_comparison_summary.json"
COMPARISON_TABLE_RELATIVE_PATH = "results/comparisons/backend_comparison_table.md"


def _metric_cell(value: object, *, integer: bool = False) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if integer:
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_backend_comparison_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Configuration | Status | Requested Backend | Requested LLM | Resolved Backend | Resolved LLM | Questions | Answerable | Unanswerable | Recall@3 | MRR@3 | Faithfulness | Citation Validity | Abstention Accuracy | False Abstention | Avg Latency (ms) | Reason |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("label", "unknown")),
                    str(row.get("status", "unknown")),
                    str(row.get("requested_backend", "n/a")),
                    str(row.get("requested_llm", "n/a")),
                    _metric_cell(row.get("resolved_backend")),
                    _metric_cell(row.get("resolved_llm")),
                    _metric_cell(row.get("questions"), integer=True),
                    _metric_cell(row.get("answerable_questions"), integer=True),
                    _metric_cell(row.get("unanswerable_questions"), integer=True),
                    _metric_cell(row.get("retrieval_recall_at_3")),
                    _metric_cell(row.get("mrr_at_3")),
                    _metric_cell(row.get("faithfulness")),
                    _metric_cell(row.get("citation_validity")),
                    _metric_cell(row.get("abstention_accuracy")),
                    _metric_cell(row.get("false_abstention_rate")),
                    _metric_cell(row.get("average_latency_ms")),
                    str(row.get("reason", "")) or "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_backend_comparison_section(summary_payload: dict[str, object] | None) -> str:
    if not summary_payload or not summary_payload.get("configs"):
        return (
            "_Backend comparison results have not been generated yet. "
            "Run `python scripts/run_backend_comparison.py --offline-only` to populate this "
            "section with real artifacts from `results/comparisons/`._"
        )

    generated_at = summary_payload.get("generated_at", "unknown time")
    table = render_backend_comparison_table(list(summary_payload["configs"]))
    lines = [
        "Auto-generated from real comparison artifacts under "
        f"`{COMPARISON_SUMMARY_RELATIVE_PATH}` and `{COMPARISON_TABLE_RELATIVE_PATH}`.",
        f"Latest comparison run: `{generated_at}`.",
        "",
        table,
    ]
    return "\n".join(lines)


def write_backend_comparison_table(output_path: Path, summary_payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_backend_comparison_section(summary_payload), encoding="utf-8")


def _replace_marked_section(text: str, replacement: str) -> str:
    start_index = text.find(BACKEND_COMPARISON_START)
    end_index = text.find(BACKEND_COMPARISON_END)
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise RagFaqError("backend comparison markers were not found")
    end_index += len(BACKEND_COMPARISON_END)
    return (
        text[:start_index]
        + BACKEND_COMPARISON_START
        + "\n"
        + replacement.strip()
        + "\n"
        + BACKEND_COMPARISON_END
        + text[end_index:]
    )


def sync_backend_comparison_docs(
    paths: PathConfig | None = None,
    summary_payload: dict[str, object] | None = None,
) -> None:
    paths = paths or get_paths()
    if summary_payload is None:
        loaded = load_json(paths.backend_comparison_summary_path, default=None)
        summary_payload = loaded if isinstance(loaded, dict) and loaded else None
    replacement = render_backend_comparison_section(summary_payload)
    for target_path in (paths.readme_path, paths.project_report_path):
        original = target_path.read_text(encoding="utf-8")
        updated = _replace_marked_section(original, replacement)
        target_path.write_text(updated, encoding="utf-8")
