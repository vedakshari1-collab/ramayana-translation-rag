from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from src.utils import safe_preview, word_count_english

ENTITY_GROUPS = ("people", "places", "concepts")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _clean_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value or "")]


def _chunk_words(chunk: dict[str, Any]) -> int:
    return int(chunk.get("word_count") or word_count_english(chunk.get("english_translation", "")))


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_chunk_entities(chunk: dict[str, Any], groups: tuple[str, ...] = ENTITY_GROUPS) -> list[str]:
    entities = chunk.get("entities") or {}
    found: list[str] = []
    if not isinstance(entities, dict):
        return found

    for group in groups:
        values = entities.get(group) or []
        if isinstance(values, str):
            values = [values]
        found.extend(_clean_label(value) for value in values if _clean_label(value))
    return found


def build_entity_facets(chunks: list[dict[str, Any]], limit: int = 12) -> dict[str, list[dict[str, Any]]]:
    facets: dict[str, list[dict[str, Any]]] = {}
    for group in ENTITY_GROUPS:
        counter: Counter[str] = Counter()
        for chunk in chunks:
            entities = chunk.get("entities") or {}
            values = entities.get(group) if isinstance(entities, dict) else []
            if isinstance(values, str):
                values = [values]
            counter.update(_clean_label(value) for value in values or [] if _clean_label(value))
        facets[group] = [{"name": name, "count": count} for name, count in counter.most_common(limit)]
    return facets


def build_kanda_summary(chunks: list[dict[str, Any]], keyword_limit: int = 6) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        kanda = _clean_label(chunk.get("kanda")) or "Unmapped"
        chapter = _safe_int(chunk.get("chapter_number"))
        page_start = _safe_int(chunk.get("page_start"))
        page_end = _safe_int(chunk.get("page_end"))
        sequence = _safe_int(chunk.get("sequence_number")) or 0

        if kanda not in grouped:
            grouped[kanda] = {
                "kanda": kanda,
                "chunk_count": 0,
                "chapters": set(),
                "pages": [],
                "word_count": 0,
                "first_sequence": sequence,
                "keywords": Counter(),
                "entities": Counter(),
                "opening_preview": "",
            }

        row = grouped[kanda]
        row["chunk_count"] += 1
        row["word_count"] += _chunk_words(chunk)
        row["first_sequence"] = min(row["first_sequence"], sequence)
        if chapter is not None:
            row["chapters"].add(chapter)
        if page_start is not None:
            row["pages"].append(page_start)
        if page_end is not None:
            row["pages"].append(page_end)
        row["keywords"].update(_clean_label(keyword) for keyword in chunk.get("keywords") or [] if _clean_label(keyword))
        row["entities"].update(iter_chunk_entities(chunk))
        if not row["opening_preview"]:
            row["opening_preview"] = safe_preview(chunk.get("summary") or chunk.get("english_translation", ""), 220)

    summary: list[dict[str, Any]] = []
    for row in grouped.values():
        pages = row["pages"]
        chunk_count = row["chunk_count"]
        summary.append(
            {
                "kanda": row["kanda"],
                "chunk_count": chunk_count,
                "chapter_count": len(row["chapters"]),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "word_count": row["word_count"],
                "average_chunk_words": round(row["word_count"] / chunk_count, 1) if chunk_count else 0,
                "top_keywords": [name for name, _count in row["keywords"].most_common(keyword_limit)],
                "top_entities": [name for name, _count in row["entities"].most_common(keyword_limit)],
                "opening_preview": row["opening_preview"],
                "_first_sequence": row["first_sequence"],
            }
        )

    summary.sort(key=lambda item: (item["_first_sequence"], item["kanda"]))
    for item in summary:
        item.pop("_first_sequence", None)
    return summary


def translation_health(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    word_counts = [_chunk_words(chunk) for chunk in chunks]
    missing_summary = sum(1 for chunk in chunks if not _clean_label(chunk.get("summary")))
    missing_keywords = sum(1 for chunk in chunks if not chunk.get("keywords"))
    missing_entities = sum(1 for chunk in chunks if not iter_chunk_entities(chunk))
    low_word_chunks = sum(1 for count in word_counts if count < 120)
    high_word_chunks = sum(1 for count in word_counts if count > 800)

    return {
        "chunk_count": len(chunks),
        "word_count": sum(word_counts),
        "average_words": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
        "low_word_chunks": low_word_chunks,
        "high_word_chunks": high_word_chunks,
        "missing_summary": missing_summary,
        "missing_keywords": missing_keywords,
        "missing_entities": missing_entities,
    }


def _score_chunk(chunk: dict[str, Any], query: str) -> int:
    tokens = set(_tokens(query))
    if not tokens:
        return 0

    score = 0
    phrase = query.strip().lower()
    weighted_fields = {
        "chunk_id": 1,
        "kanda": 6,
        "chapter_title": 6,
        "summary": 4,
        "notes": 2,
        "english_translation": 1,
    }

    for field, weight in weighted_fields.items():
        field_tokens = set(_tokens(str(chunk.get(field) or "")))
        score += weight * len(tokens & field_tokens)

    keyword_tokens = set(_tokens(" ".join(str(item) for item in chunk.get("keywords") or [])))
    entity_tokens = set(_tokens(" ".join(iter_chunk_entities(chunk))))
    score += 8 * len(tokens & keyword_tokens)
    score += 7 * len(tokens & entity_tokens)

    combined = " ".join(
        str(chunk.get(field) or "")
        for field in ["summary", "english_translation", "notes", "chapter_title", "kanda"]
    ).lower()
    if len(phrase) > 2 and phrase in combined:
        score += 12
    return score


def _entity_matches(chunk: dict[str, Any], entity: str | None) -> bool:
    if not entity or entity == "All":
        return True
    wanted = entity.lower()
    return any(wanted == value.lower() for value in iter_chunk_entities(chunk))


def ranked_search_chunks(
    chunks: list[dict[str, Any]],
    query: str = "",
    *,
    kanda: str | None = None,
    chapter: int | str | None = None,
    entity: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    requested_chapter = None if chapter in (None, "", "All") else _safe_int(chapter)
    query = query or ""
    results: list[dict[str, Any]] = []

    for chunk in chunks:
        if kanda and kanda != "All" and chunk.get("kanda") != kanda:
            continue
        if requested_chapter is not None and _safe_int(chunk.get("chapter_number")) != requested_chapter:
            continue
        if not _entity_matches(chunk, entity):
            continue

        score = _score_chunk(chunk, query)
        if query.strip() and score == 0:
            continue
        ranked = dict(chunk)
        ranked["_search_score"] = score
        ranked["_preview"] = safe_preview(chunk.get("english_translation", ""), 260)
        results.append(ranked)

    if query.strip():
        results.sort(key=lambda item: (-item["_search_score"], _safe_int(item.get("sequence_number")) or 0))
    else:
        results.sort(key=lambda item: _safe_int(item.get("sequence_number")) or 0)

    if max_results is not None:
        return results[:max_results]
    return results
