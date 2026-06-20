from __future__ import annotations

from src.explore_chunks import build_entity_facets, build_kanda_summary, ranked_search_chunks, translation_health


def sample_chunks():
    return [
        {
            "chunk_id": "bala_kanda_chapter_001_chunk_001",
            "kanda": "Bala Kanda",
            "chapter_number": 1,
            "chapter_title": "The Birth of Rama",
            "page_start": 10,
            "page_end": 11,
            "sequence_number": 1,
            "english_translation": "Rama is born in Ayodhya. The city celebrates the royal birth.",
            "summary": "Rama is born in Ayodhya.",
            "keywords": ["Rama", "Ayodhya"],
            "entities": {"people": ["Rama"], "places": ["Ayodhya"], "concepts": ["Dharma"]},
            "word_count": 12,
        },
        {
            "chunk_id": "ayodhya_kanda_chapter_002_chunk_002",
            "kanda": "Ayodhya Kanda",
            "chapter_number": 2,
            "chapter_title": "Exile",
            "page_start": 12,
            "page_end": 13,
            "sequence_number": 2,
            "english_translation": "Sita and Lakshmana follow Rama into the forest with devotion.",
            "summary": "The exile begins.",
            "keywords": ["Exile", "Forest"],
            "entities": {"people": ["Rama", "Sita", "Lakshmana"], "places": ["Forest"], "concepts": ["Devotion"]},
            "word_count": 10,
        },
    ]


def test_build_kanda_summary_groups_chunk_statistics():
    summary = build_kanda_summary(sample_chunks())

    assert [row["kanda"] for row in summary] == ["Bala Kanda", "Ayodhya Kanda"]
    assert summary[0]["chunk_count"] == 1
    assert summary[0]["chapter_count"] == 1
    assert summary[0]["page_start"] == 10
    assert "Rama" in summary[0]["top_entities"]


def test_ranked_search_prioritizes_entities_and_keywords():
    results = ranked_search_chunks(sample_chunks(), "Sita")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "ayodhya_kanda_chapter_002_chunk_002"
    assert results[0]["_search_score"] > 0
    assert results[0]["_preview"]


def test_ranked_search_filters_by_entity():
    results = ranked_search_chunks(sample_chunks(), entity="Lakshmana")

    assert [row["chunk_id"] for row in results] == ["ayodhya_kanda_chapter_002_chunk_002"]


def test_entity_facets_and_health_are_summarized():
    facets = build_entity_facets(sample_chunks())
    health = translation_health(sample_chunks())

    assert facets["people"][0] == {"name": "Rama", "count": 2}
    assert health["chunk_count"] == 2
    assert health["low_word_chunks"] == 2
