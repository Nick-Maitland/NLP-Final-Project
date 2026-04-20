from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable


class RagFaqError(RuntimeError):
    """Raised when the CLI should surface a user-facing project error."""


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "does",
    "do",
    "each",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "more",
    "of",
    "on",
    "or",
    "same",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "use",
    "uses",
    "using",
    "what",
    "when",
    "which",
    "why",
    "with",
}


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def sentence_split(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def make_source_id(path: Path) -> str:
    return slugify(path.stem)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: object | None = None) -> object:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_markdown_or_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
        text = "\n".join(lines)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    return normalize_text(text)


def read_html_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        from bs4 import BeautifulSoup
    except Exception:
        text = re.sub(r"<[^>]+>", " ", text)
        return normalize_text(text)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    return normalize_text(soup.get_text(separator="\n"))


def read_supported_text(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return read_markdown_or_text(path)
    if path.suffix.lower() in {".html", ".htm"}:
        return read_html_text(path)
    raise RagFaqError(f"Unsupported knowledge-base file type: {path.name}")


def extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned
    return path.stem.replace("_", " ").title()


def format_sources(source_ids: list[str]) -> str:
    if not source_ids:
        return "none"
    return ", ".join(source_ids)
