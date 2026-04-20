from __future__ import annotations

import csv
import hashlib
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


def stable_text_hash(text: str) -> str:
    normalized = normalize_text(text).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.replace("\ufeff", "").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    metadata: dict[str, str] = {}
    end_index = None
    for index in range(1, len(lines)):
        line = lines[index].strip()
        if line == "---":
            end_index = index
            break
        if ":" not in lines[index]:
            continue
        key, value = lines[index].split(":", 1)
        metadata[key.strip().lower()] = value.strip()

    if end_index is None:
        return {}, text
    return metadata, "\n".join(lines[end_index + 1 :])


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def token_signature(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(content_tokens(text))))


def jaccard_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


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


def infer_topic_from_identifier(identifier: str) -> str:
    topic = slugify(identifier)
    if topic.startswith("faq_"):
        topic = topic[4:]
    topic = re.sub(r"_\d+$", "", topic)
    return topic or "general"


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
    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    lines = body.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
        body = "\n".join(lines)
    text = re.sub(r"^\s*#+\s*", "", body, flags=re.MULTILINE)
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


def extract_title(path: Path, text: str, metadata: dict[str, str] | None = None) -> str:
    metadata = metadata or {}
    if metadata.get("title"):
        return metadata["title"]
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned
    return path.stem.replace("_", " ").title()


def read_text_with_metadata(path: Path) -> tuple[str, dict[str, str], str]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw_text)
        title = metadata.get("title")
        if not title:
            for line in body.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    title = stripped.lstrip("#").strip()
                    break
        text = read_markdown_or_text(path)
        return text, metadata, title or path.stem.replace("_", " ").title()
    if suffix in {".html", ".htm"}:
        text = read_html_text(path)
        return text, {}, extract_title(path, text)
    raise RagFaqError(f"Unsupported knowledge-base file type: {path.name}")


def format_sources(source_ids: list[str]) -> str:
    if not source_ids:
        return "none"
    return ", ".join(source_ids)
