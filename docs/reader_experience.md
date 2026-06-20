# Reader Experience Notes

This project now treats the final RAG chunks as a browsable source map, not only as export files. The Streamlit app adds a Reading Map and ranked explorer so reviewers can inspect narrative coverage before using the data in a retrieval system.

## Review Goals

- Show whether each kanda has enough chunk, chapter, page, and word coverage.
- Surface recurring people, places, and concepts as quick review facets.
- Make search results easier to audit by ranking direct keyword and entity hits above incidental text matches.
- Keep the review workflow deterministic; no extra LLM calls are made after the reviewed translation JSONL exists.

## App Changes

- The dashboard now opens with kanda coverage and entity facets when chunks are available.
- The Reading Map page summarizes chunk health, kanda coverage, and top entities.
- The Chunk Explorer ranks matches across metadata, summaries, translation text, keywords, notes, and entities.
- Filtered search results can be exported as JSON for focused review or downstream experiments.

## Data Quality Signals

The exploration layer flags gaps that matter for retrieval quality:

- very short chunks that may not have enough context,
- very long chunks that may be awkward to embed,
- missing summaries,
- missing keywords,
- missing entity metadata.

These signals are review aids, not blockers. The validation report remains the source of truth for required schema checks.
