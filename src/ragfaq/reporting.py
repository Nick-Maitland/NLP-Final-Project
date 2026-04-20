from __future__ import annotations

from pathlib import Path

from .schemas import EvaluationRow


def generate_failure_report(results: list[EvaluationRow], output_path: Path) -> None:
    total = len(results)
    average_recall = sum(result.recall_at_3 for result in results) / max(total, 1)
    average_faithfulness = sum(result.faithfulness_score for result in results) / max(total, 1)
    failures = [
        result
        for result in results
        if result.recall_at_3 < 1.0 or result.faithfulness_score < 1.0
    ]

    lines = [
        "# Failure Case Report",
        "",
        "## Summary",
        "",
        f"- Evaluated questions: {total}",
        f"- Average Recall@3: {average_recall:.2f}",
        f"- Average faithfulness: {average_faithfulness:.2f}",
        f"- Questions with issues: {len(failures)}",
        "",
        "## Failure Cases",
        "",
    ]

    if not failures:
        lines.append("- No failure cases were detected in the current evaluation run.")
    else:
        for result in failures:
            lines.extend(
                [
                    f"### {result.question_id}: {result.question}",
                    f"- Expected source: `{result.expected_source_id}`",
                    f"- Retrieved sources: `{result.retrieved_source_ids or 'none'}`",
                    f"- Recall@3: {result.recall_at_3:.2f}",
                    f"- Faithfulness: {result.faithfulness_score:.2f}",
                    f"- Notes: {result.notes or 'n/a'}",
                    f"- Generated answer: {result.generated_answer}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Improvement Priorities",
            "",
            "- Expand or sharpen knowledge-base passages where expected sources are missed.",
            "- Adjust chunking or lexical weighting when semantically relevant passages rank too low.",
            "- Keep offline answers extractive so unsupported claims do not reduce faithfulness.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
