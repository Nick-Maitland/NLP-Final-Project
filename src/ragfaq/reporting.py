from __future__ import annotations

import statistics
from pathlib import Path

from .schemas import EvaluationRow


def _split_values(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def summarize_results(results: list[EvaluationRow]) -> dict[str, object]:
    answerable = [result for result in results if result.answerable]
    unanswerable = [result for result in results if not result.answerable]
    recall_values = [
        result.retrieval_recall_at_3 for result in answerable if result.retrieval_recall_at_3 is not None
    ]
    mrr_values = [
        result.reciprocal_rank for result in answerable if result.reciprocal_rank is not None
    ]
    faithfulness_values = [result.faithfulness_score for result in results]
    latency_values = [result.latency_ms for result in results]
    citation_valid_values = [1.0 if result.citation_valid else 0.0 for result in results]
    abstention_values = [
        1.0 if result.abstention_correct else 0.0 for result in unanswerable
    ]
    false_abstention_values = [1.0 if result.abstained else 0.0 for result in answerable]

    per_topic: dict[str, dict[str, float | int]] = {}
    for result in results:
        topics = _split_values(result.expected_topic) or ["unknown"]
        for topic in topics:
            bucket = per_topic.setdefault(
                topic,
                {
                    "question_count": 0,
                    "answerable_count": 0,
                    "unanswerable_count": 0,
                    "retrieval_recall_at_3_avg": 0.0,
                    "faithfulness_avg": 0.0,
                },
            )
            bucket["question_count"] += 1
            if result.answerable:
                bucket["answerable_count"] += 1
                if result.retrieval_recall_at_3 is not None:
                    bucket["retrieval_recall_at_3_avg"] += result.retrieval_recall_at_3
            else:
                bucket["unanswerable_count"] += 1
            bucket["faithfulness_avg"] += result.faithfulness_score

    for topic, bucket in per_topic.items():
        question_count = max(int(bucket["question_count"]), 1)
        answerable_count = max(int(bucket["answerable_count"]), 1)
        bucket["faithfulness_avg"] = round(float(bucket["faithfulness_avg"]) / question_count, 2)
        if int(bucket["answerable_count"]) > 0:
            bucket["retrieval_recall_at_3_avg"] = round(
                float(bucket["retrieval_recall_at_3_avg"]) / answerable_count,
                2,
            )
        else:
            bucket["retrieval_recall_at_3_avg"] = None

    return {
        "question_count": len(results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "retrieval_recall_at_3_answerable": round(sum(recall_values) / max(len(recall_values), 1), 2),
        "mrr_at_3_answerable": round(sum(mrr_values) / max(len(mrr_values), 1), 2),
        "faithfulness_avg": round(sum(faithfulness_values) / max(len(faithfulness_values), 1), 2),
        "citation_valid_rate": round(
            sum(citation_valid_values) / max(len(citation_valid_values), 1),
            2,
        ),
        "abstention_accuracy_unanswerable": round(
            sum(abstention_values) / max(len(abstention_values), 1),
            2,
        ) if unanswerable else None,
        "false_abstention_rate_answerable": round(
            sum(false_abstention_values) / max(len(false_abstention_values), 1),
            2,
        ) if answerable else None,
        "avg_latency_ms": round(sum(latency_values) / max(len(latency_values), 1), 2),
        "median_latency_ms": round(statistics.median(latency_values), 2) if latency_values else 0.0,
        "per_topic": dict(sorted(per_topic.items())),
    }


def _severity(result: EvaluationRow) -> float:
    severity = 0.0
    if result.answerable:
        severity += (1.0 - (result.retrieval_recall_at_3 or 0.0)) * 3.0
        severity += (1.0 - (result.reciprocal_rank or 0.0)) * 1.5
    if not result.citation_valid:
        severity += 1.5
    if not result.abstention_correct:
        severity += 2.0
    severity += (1.0 - result.faithfulness_score) * 2.5
    return severity


def classify_failure(result: EvaluationRow) -> str:
    if not result.answerable and not result.abstention_correct:
        return "generation"
    if result.answerable and result.abstained:
        if (result.retrieval_recall_at_3 or 0.0) < 1.0:
            return "retrieval"
        return "generation"
    if result.answerable and (result.retrieval_recall_at_3 or 0.0) < 0.5:
        return "retrieval"
    if result.answerable and result.faithfulness_score < 0.8:
        return "generation"
    if not result.citation_valid:
        return "generation"
    return "knowledge_base_coverage"


def _proposed_fix(failure_type: str) -> str:
    if failure_type == "retrieval":
        return "Tune lexical weighting, add paraphrase coverage, or improve multi-hop candidate fusion."
    if failure_type == "generation":
        return "Tighten extractive sentence selection and citation filtering so unsupported text is excluded."
    return "Expand the knowledge base with clearer passages for this concept or domain."


def _issue_reason(result: EvaluationRow) -> str:
    reasons = []
    if result.answerable and (result.retrieval_recall_at_3 or 0.0) < 1.0:
        reasons.append("expected evidence was not fully retrieved in the top 3")
    if result.faithfulness_score < 1.0:
        reasons.append("the answer was only partially grounded in retrieved context")
    if not result.citation_valid:
        reasons.append("citations were missing or structurally invalid")
    if not result.abstention_correct:
        reasons.append("the abstention decision did not match answerability")
    return "; ".join(reasons) or "borderline example with weaker-than-average support"


def generate_failure_report(results: list[EvaluationRow], output_path: Path) -> None:
    summary = summarize_results(results)
    ranked = sorted(results, key=_severity, reverse=True)
    examples = ranked[: max(3, min(5, len(ranked)))]

    lines = [
        "# Failure Case Report",
        "",
        "## Summary",
        "",
        f"- Evaluated questions: {summary['question_count']}",
        f"- Answerable questions: {summary['answerable_count']}",
        f"- Unanswerable questions: {summary['unanswerable_count']}",
        f"- Recall@3 on answerable questions: {summary['retrieval_recall_at_3_answerable']:.2f}",
        f"- MRR@3 on answerable questions: {summary['mrr_at_3_answerable']:.2f}",
        f"- Faithfulness average: {summary['faithfulness_avg']:.2f}",
        f"- Citation valid rate: {summary['citation_valid_rate']:.2f}",
        (
            "- False abstention rate (answerable): "
            f"{summary['false_abstention_rate_answerable']:.2f}"
        ),
        "",
        "## Concrete Weak Examples",
        "",
    ]

    for result in examples:
        failure_type = classify_failure(result)
        lines.extend(
            [
                f"### {result.question_id}: {result.question}",
                f"- Failure type: `{failure_type}`",
                f"- Expected source/topic: `{result.expected_source_id or 'none'}` / `{result.expected_topic}`",
                f"- Retrieved sources: `{result.retrieved_source_ids or 'none'}`",
                f"- Retrieved chunks: {result.retrieved_chunk_summaries or 'none'}",
                f"- Answer: {result.answer or 'none'}",
                f"- Why it failed: {_issue_reason(result)}",
                f"- Proposed fix: {_proposed_fix(failure_type)}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_evaluation_report(
    results: list[EvaluationRow],
    summary: dict[str, object],
    output_path: Path,
) -> None:
    multi_hop = [
        result
        for result in results
        if len(_split_values(result.expected_source_id)) > 1 or "multi-hop" in result.notes
    ]
    paraphrase = [result for result in results if "paraphrase" in result.notes]
    unanswerable_failures = [
        result for result in results if not result.answerable and not result.abstention_correct
    ]
    multi_hop_answerable = [result for result in multi_hop if result.answerable]
    multi_hop_recall = None
    if multi_hop_answerable:
        recall_values = [
            result.retrieval_recall_at_3
            for result in multi_hop_answerable
            if result.retrieval_recall_at_3 is not None
        ]
        if recall_values:
            multi_hop_recall = round(sum(recall_values) / len(recall_values), 2)

    weakest_topics = []
    for topic, metrics in summary["per_topic"].items():
        if metrics["answerable_count"] > 0 and metrics["retrieval_recall_at_3_avg"] is not None:
            weakest_topics.append((topic, metrics["retrieval_recall_at_3_avg"]))
    weakest_topics = sorted(weakest_topics, key=lambda item: item[1])[:3]

    lines = [
        "# Evaluation Report",
        "",
        "## Summary Metrics",
        "",
        f"- Questions: {summary['question_count']}",
        f"- Answerable: {summary['answerable_count']}",
        f"- Unanswerable: {summary['unanswerable_count']}",
        f"- Retrieval Recall@3 (answerable): {summary['retrieval_recall_at_3_answerable']:.2f}",
        f"- MRR@3 (answerable): {summary['mrr_at_3_answerable']:.2f}",
        f"- Faithfulness average: {summary['faithfulness_avg']:.2f}",
        f"- Citation valid rate: {summary['citation_valid_rate']:.2f}",
        f"- Abstention accuracy (unanswerable): {summary['abstention_accuracy_unanswerable'] if summary['abstention_accuracy_unanswerable'] is not None else 'n/a'}",
        f"- False abstention rate (answerable): {summary['false_abstention_rate_answerable'] if summary['false_abstention_rate_answerable'] is not None else 'n/a'}",
        f"- Average latency (ms): {summary['avg_latency_ms']:.2f}",
        f"- Median latency (ms): {summary['median_latency_ms']:.2f}",
        "",
        "## Topic Breakdown",
        "",
    ]

    for topic, metrics in summary["per_topic"].items():
        lines.append(
            f"- `{topic}`: questions={metrics['question_count']}, "
            f"answerable={metrics['answerable_count']}, "
            f"recall@3={metrics['retrieval_recall_at_3_avg']}, "
            f"faithfulness={metrics['faithfulness_avg']}"
        )

    lines.extend(
        [
            "",
            "## Analysis",
            "",
            f"- Multi-hop questions: {len(multi_hop)}",
            f"- Multi-hop Recall@3 (answerable): {multi_hop_recall if multi_hop_recall is not None else 'n/a'}",
            f"- Paraphrased questions: {len(paraphrase)}",
            f"- Unanswerable questions answered incorrectly: {len(unanswerable_failures)}",
            f"- Citation-valid answers: {summary['citation_valid_rate']:.2f}",
            f"- False abstentions on answerable questions: {summary['false_abstention_rate_answerable'] if summary['false_abstention_rate_answerable'] is not None else 'n/a'}",
            f"- Weakest answerable topics by recall: "
            + (
                ", ".join(f"{topic} ({score:.2f})" for topic, score in weakest_topics)
                if weakest_topics
                else "n/a"
            ),
            "- Weak areas are reflected directly in the concrete examples below rather than being smoothed away.",
            "- Imperfect results are expected on paraphrase and multi-hop rows because they stress retrieval coverage and grounded composition.",
            "",
            "## Weak Examples",
            "",
        ]
    )

    ranked = sorted(results, key=_severity, reverse=True)[:5]
    for result in ranked:
        lines.extend(
            [
                f"### {result.question_id}: {result.question}",
                f"- Notes: {result.notes or 'n/a'}",
                f"- Recall@3: {result.retrieval_recall_at_3 if result.retrieval_recall_at_3 is not None else 'n/a'}",
                f"- Reciprocal rank: {result.reciprocal_rank if result.reciprocal_rank is not None else 'n/a'}",
                f"- Faithfulness: {result.faithfulness_score:.2f}",
                f"- Citation valid: {str(result.citation_valid).lower()}",
                f"- Abstention correct: {str(result.abstention_correct).lower()}",
                f"- Retrieved chunks: {result.retrieved_chunk_summaries or 'none'}",
                f"- Answer: {result.answer or 'none'}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
