from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.chunking import chunk_documents
from ragfaq.config import (
    DEFAULT_TOP_K,
    ensure_runtime_directories,
    get_paths,
    get_runtime_availability,
)
from ragfaq.evaluation import run_evaluation
from ragfaq.generation import answer_question
from ragfaq.ingest import load_documents
from ragfaq.retrievers import inspect_index_state, maybe_build_indexes, retrieve
from ragfaq.schemas import BackendMode, LlmMode
from ragfaq.utils import RagFaqError, format_sources


COMPATIBILITY_TEXT = (
    "Compatibility aliases remain supported: "
    "--build-index, --ask \"QUESTION\", --evaluate, and --smoke-test --offline."
)


def _choices(enum_cls) -> list[str]:
    return [item.value for item in enum_cls]


def _legacy_to_subcommand(argv: list[str]) -> list[str]:
    args = list(argv)
    if not args:
        return args
    if args[0] in {"build", "ask", "evaluate", "inspect-kb", "demo"}:
        return args

    if "--build-index" in args:
        args.remove("--build-index")
        return ["build", *args]

    if "--ask" in args:
        index = args.index("--ask")
        try:
            question = args[index + 1]
        except IndexError as exc:
            raise RagFaqError("Legacy --ask mode requires a question string.") from exc
        remaining = args[:index] + args[index + 2 :]
        return ["ask", "--question", question, *remaining]

    if "--evaluate" in args:
        args.remove("--evaluate")
        return ["evaluate", *args]

    if "--smoke-test" in args:
        args.remove("--smoke-test")
        rewritten = ["demo", *args]
        if "--backend" not in rewritten:
            rewritten.extend(["--backend", BackendMode.TFIDF.value])
        if "--llm" not in rewritten and "--offline" not in rewritten:
            rewritten.extend(["--llm", LlmMode.OFFLINE.value])
        return rewritten

    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Course Project 10 RAG FAQ system.",
        epilog=COMPATIBILITY_TEXT,
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        choices=_choices(BackendMode),
        default=BackendMode.AUTO.value,
        help="Retrieval backend to use.",
    )
    common.add_argument(
        "--llm",
        choices=_choices(LlmMode),
        default=LlmMode.AUTO.value,
        help="LLM mode to use.",
    )
    common.add_argument(
        "--offline",
        action="store_true",
        help="Convenience alias for --llm offline.",
    )
    common.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )

    subparsers.add_parser("build", parents=[common], help="Build indexes from the knowledge base.")

    ask_parser = subparsers.add_parser(
        "ask",
        parents=[common],
        help="Ask a question against the indexed knowledge base.",
    )
    ask_parser.add_argument("--question", required=True, help="Question to answer.")

    subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Run the 30-question evaluation and refresh the report.",
    )
    subparsers.add_parser(
        "inspect-kb",
        parents=[common],
        help="Inspect repo-local knowledge-base files and index availability.",
    )
    demo_parser = subparsers.add_parser(
        "demo",
        parents=[common],
        help="Run an offline-safe smoke demo.",
    )
    demo_parser.add_argument(
        "--question",
        default="What is self-attention?",
        help="Optional demo question.",
    )

    return parser


def _resolve_requested_modes(args: argparse.Namespace) -> tuple[BackendMode, LlmMode]:
    backend = BackendMode(args.backend)
    llm = LlmMode.OFFLINE if args.offline else LlmMode(args.llm)
    return backend, llm


def _load_docs_and_chunks():
    paths = ensure_runtime_directories(get_paths())
    documents = load_documents(paths)
    if not documents:
        raise RagFaqError(
            "No knowledge-base files were found. Populate the root knowledge_base/ folder first."
        )
    chunks = chunk_documents(documents)
    return paths, documents, chunks


def command_build(args: argparse.Namespace) -> int:
    requested_backend, _ = _resolve_requested_modes(args)
    paths, documents, chunks = _load_docs_and_chunks()
    summary = maybe_build_indexes(chunks, requested_backend=requested_backend, paths=paths)
    print(f"Knowledge-base documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    lexical_summary = summary["lexical_index"]
    print(f"Lexical index: ready at {lexical_summary['path']}")
    dense_summary = summary["dense_index"]
    if dense_summary.get("built"):
        print(f"Dense index: built with {dense_summary['document_count']} stored chunks")
    else:
        print(f"Dense index: skipped ({dense_summary.get('reason', 'not requested')})")
    return 0


def command_inspect_kb(args: argparse.Namespace) -> int:
    paths, documents, chunks = _load_docs_and_chunks()
    availability = get_runtime_availability()
    index_state = inspect_index_state(paths)
    print(f"Knowledge-base directory: {paths.knowledge_base_dir}")
    print(f"Source documents: {len(documents)}")
    print(f"Chunk count (computed): {len(chunks)}")
    print(f"Source IDs: {format_sources([document.source_id for document in documents])}")
    print(f"Lexical index ready: {index_state['lexical_index_ready']}")
    print(f"Dense index ready: {index_state['dense_index_ready']}")
    print(f"Chroma SDK available: {availability.chromadb.available}")
    print(f"Sentence-transformers available: {availability.sentence_transformers.available}")
    if not availability.sentence_transformers.available:
        print(f"Dense availability reason: {index_state['dense_runtime_reason']}")
    print(f"OpenAI SDK available: {availability.openai_sdk.available}")
    print(f"OPENAI_API_KEY present: {availability.openai_key_available}")
    return 0


def command_ask(args: argparse.Namespace) -> int:
    requested_backend, requested_llm = _resolve_requested_modes(args)
    paths = ensure_runtime_directories(get_paths())
    retrieved_chunks, resolved_backend = retrieve(
        question=args.question,
        requested_backend=requested_backend,
        top_k=args.top_k,
        paths=paths,
    )
    answer = answer_question(
        question=args.question,
        retrieved_chunks=retrieved_chunks,
        requested_llm=requested_llm,
        resolved_backend=resolved_backend,
    )
    print(f"Question: {args.question}")
    print(f"Resolved backend: {answer.resolved_backend.value}")
    print(f"Resolved llm: {answer.resolved_llm.value}")
    print(f"Sources: {format_sources(answer.sources)}")
    print("")
    print(answer.answer)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    requested_backend, requested_llm = _resolve_requested_modes(args)
    paths, _, chunks = _load_docs_and_chunks()
    maybe_build_indexes(chunks, requested_backend=requested_backend, paths=paths)
    results = run_evaluation(
        requested_backend=requested_backend,
        requested_llm=requested_llm,
        paths=paths,
        top_k=args.top_k,
    )
    average_recall = sum(result.recall_at_3 for result in results) / max(len(results), 1)
    average_faithfulness = sum(result.faithfulness_score for result in results) / max(
        len(results), 1
    )
    print(f"Evaluated rows: {len(results)}")
    print(f"Average Recall@3: {average_recall:.2f}")
    print(f"Average faithfulness: {average_faithfulness:.2f}")
    print(f"Updated CSV: {paths.test_questions_path}")
    print(f"Updated report: {paths.failure_report_path}")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    requested_backend, requested_llm = _resolve_requested_modes(args)
    paths, _, chunks = _load_docs_and_chunks()
    if not paths.lexical_index_path.exists():
        maybe_build_indexes(chunks, requested_backend=BackendMode.TFIDF, paths=paths)
    demo_backend = requested_backend
    demo_llm = requested_llm
    retrieved_chunks, resolved_backend = retrieve(
        question=args.question,
        requested_backend=demo_backend,
        top_k=args.top_k,
        paths=paths,
    )
    answer = answer_question(
        question=args.question,
        retrieved_chunks=retrieved_chunks,
        requested_llm=demo_llm,
        resolved_backend=resolved_backend,
    )
    print("Demo mode")
    print(f"Question: {args.question}")
    print(f"Resolved backend: {answer.resolved_backend.value}")
    print(f"Resolved llm: {answer.resolved_llm.value}")
    print(f"Sources: {format_sources(answer.sources)}")
    print("")
    print(answer.answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    rewritten = _legacy_to_subcommand(argv)
    args = parser.parse_args(rewritten)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "build": command_build,
        "ask": command_ask,
        "evaluate": command_evaluate,
        "inspect-kb": command_inspect_kb,
        "demo": command_demo,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RagFaqError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
