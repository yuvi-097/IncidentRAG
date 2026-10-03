"""End-to-end ingestion over the full corpus: provenance, metadata, persistence, filtering."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import Document, DocumentChunk, Incident
from app.rag.chunking import ChunkingConfig, count_tokens
from app.rag.ingestion import IngestionPipeline
from app.rag.ingestion.models import SOURCE_FOREIGN_KEY, ChunkRecord
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.ingestion.pipeline import source_text
from app.rag.ingestion.sources import load_sources
from app.rag.store import ChunkFilter, filtered_chunks
from app.schemas.enums import AccessLevel, ChunkingStrategy, DocumentType, SourceType
from app.security import principal_for_role
from app.synthetic.text import sha256
from tests.rag.conftest import DEFAULT_CONFIG, Ingested, sqlite_engine_with_fks


def _stored(ingested: Ingested) -> list[DocumentChunk]:
    with Session(ingested.engine) as session:
        return list(session.scalars(select(DocumentChunk).order_by(DocumentChunk.id)))


def test_every_source_record_is_ingested(ingested: Ingested) -> None:
    ds = ingested.dataset
    expected = (
        len(ds.documents)
        + len(ds.incidents)
        + len(ds.deployments)
        + len(ds.code_files)
        + len(ds.pull_requests)
    )
    assert ingested.report.total_sources == expected
    # Confidential sources are never indexed; nothing else is skipped.
    confidential = {d.id for d in ds.documents if d.access_level is AccessLevel.CONFIDENTIAL}
    assert confidential == {"DOC-0100"}
    assert [(s.source_id, s.reason) for s in ingested.report.skipped] == [
        ("DOC-0100", "confidential sources are not indexed")
    ]
    assert {c.document_id for c in ingested.chunks} == (
        {d.id for d in ds.documents if d.id not in confidential}
        | {i.id for i in ds.incidents}
        | {d.id for d in ds.deployments}
        | {f.id for f in ds.code_files}
        | {p.id for p in ds.pull_requests}
    )
    assert set(ingested.report.chunks_by_source_type) == {t.value for t in SourceType}


def test_chunks_are_persisted(ingested: Ingested) -> None:
    rows = _stored(ingested)
    assert len(rows) == len(ingested.chunks) == ingested.report.total_chunks


def test_every_chunk_traces_back_to_its_source(ingested: Ingested) -> None:
    """Re-derive each source from the database; stored offsets and hash must reproduce the chunk."""
    with ingested.engine.connect() as connection:
        sources = {(s.source_type, s.source_id): s for s in load_sources(connection)}
    texts: dict[tuple[SourceType, str], str] = {}
    for chunk in _stored(ingested):
        key = (chunk.source_type, chunk.document_id)
        assert key in sources, f"{chunk.id}: source record not found"
        if key not in texts:
            texts[key] = source_text(sources[key])
        derived = texts[key]
        assert derived[chunk.char_start : chunk.char_end] == chunk.content, chunk.id
        assert sha256(derived) == chunk.source_hash, chunk.id
        assert getattr(chunk, SOURCE_FOREIGN_KEY[chunk.source_type]) == chunk.document_id, chunk.id
        assert chunk.id == f"{chunk.document_id}#{chunk.chunk_index:03d}"


def test_document_and_code_chunks_are_verbatim_excerpts(ingested: Ingested) -> None:
    """Independent of our renderer: document and code chunks appear verbatim in the raw record."""
    raw = {d.id: d.content for d in ingested.dataset.documents} | {
        f.id: f.content for f in ingested.dataset.code_files
    }
    for chunk in ingested.chunks:
        if chunk.document_id in raw:
            assert chunk.content in raw[chunk.document_id], chunk.id


def test_no_source_content_is_lost(ingested: Ingested) -> None:
    """Chunks jointly cover every token of the cleaned source text."""
    by_source: dict[str, list[ChunkRecord]] = {}
    for chunk in ingested.chunks:
        by_source.setdefault(chunk.document_id, []).append(chunk)
    with ingested.engine.connect() as connection:
        for source in load_sources(connection):
            if source.source_id not in by_source:  # confidential: not indexed
                assert source.payload["access_level"] == AccessLevel.CONFIDENTIAL
                continue
            text_of_source = source_text(source)
            covered = [False] * len(text_of_source)
            for chunk in by_source[source.source_id]:
                covered[chunk.char_start : chunk.char_end] = [True] * (
                    chunk.char_end - chunk.char_start
                )
            missing = "".join(
                ch for ch, hit in zip(text_of_source, covered, strict=True) if not hit
            )
            assert count_tokens(missing) == 0, (source.source_id, missing[:80])


def test_metadata_matches_the_source_record(ingested: Ingested) -> None:
    ds = ingested.dataset
    incidents = {i.id: i for i in ds.incidents}
    documents = {d.id: d for d in ds.documents}
    code = {f.id: f for f in ds.code_files}
    deployments = {d.id: d for d in ds.deployments}
    for chunk in ingested.chunks:
        if chunk.source_type is SourceType.INCIDENT:
            i = incidents[chunk.document_id]
            assert (chunk.service_id, chunk.version, chunk.access_level) == (
                i.service_id,
                i.affected_version,
                i.access_level,
            )
            assert (
                chunk.timestamp == i.started_at and chunk.metadata["category"] == i.category.value
            )
        elif chunk.document_id in documents:
            d = documents[chunk.document_id]
            assert (chunk.service_id, chunk.access_level, chunk.doc_type) == (
                d.service_id,
                d.access_level,
                d.doc_type.value,
            )
            assert chunk.file_path == d.source_path and chunk.timestamp == d.updated_at
        elif chunk.source_type is SourceType.CODE:
            f = code[chunk.document_id]
            assert (chunk.file_path, chunk.service_id, chunk.access_level) == (
                f.path,
                f.service_id,
                f.access_level,
            )
            assert chunk.metadata["last_commit_sha"] == f.last_commit_sha
        elif chunk.source_type is SourceType.DEPLOYMENT:
            d = deployments[chunk.document_id]
            assert (chunk.version, chunk.service_id, chunk.timestamp) == (
                d.version,
                d.service_id,
                d.deployed_at,
            )
        assert chunk.metadata["line_start"] <= chunk.metadata["line_end"]


def test_code_chunk_lines_point_into_the_file(ingested: Ingested) -> None:
    code = {f.id: f.content.split("\n") for f in ingested.dataset.code_files}
    for chunk in ingested.chunks:
        if chunk.source_type is SourceType.CODE:
            lines = code[chunk.document_id][
                chunk.metadata["line_start"] - 1 : chunk.metadata["line_end"]
            ]
            assert chunk.content.split("\n")[0].strip() in lines[0], chunk.id


def test_pull_request_diff_chunks_carry_the_changed_file(ingested: Ingested) -> None:
    paths = {f.id: f.path for f in ingested.dataset.code_files}
    changed = {
        (p.pull_request_id, paths[p.code_file_id]) for p in ingested.dataset.pull_request_files
    }
    diff_chunks = [
        c for c in ingested.chunks if c.source_type is SourceType.PULL_REQUEST and c.file_path
    ]
    assert diff_chunks
    for chunk in diff_chunks:
        assert (chunk.document_id, chunk.file_path) in changed
        assert "```diff" in chunk.content or chunk.metadata.get("split_block")


def test_single_file_pull_request_chunk_has_that_file_path(ingested: Ingested) -> None:
    chunk = next(c for c in ingested.chunks if c.document_id == "PR-1501")
    assert chunk.file_path == "services/payment-service/payment_service/db/database.py"
    multi = [
        c
        for c in ingested.chunks
        if c.source_type is SourceType.PULL_REQUEST and c.content.count("## Diff: ") > 1
    ]
    assert multi and all(c.file_path is None for c in multi)  # ambiguous: several files


def test_restricted_code_keeps_its_label_in_pull_requests(ingested: Ingested) -> None:
    pr_chunk = next(c for c in ingested.chunks if c.document_id == "PR-1501")
    assert pr_chunk.access_level is AccessLevel.SRE  # touches payment-service code


def test_chunks_respect_the_size_limit(ingested: Ingested) -> None:
    assert max(c.token_count for c in ingested.chunks) <= DEFAULT_CONFIG.chunk_size_tokens
    assert all(c.token_count == count_tokens(c.content) for c in ingested.chunks)


def test_reingestion_is_idempotent_and_deterministic(ingested: Ingested) -> None:
    before = {c.id: c.content for c in _stored(ingested)}
    chunks, _ = IngestionPipeline(DEFAULT_CONFIG).run(ingested.engine)
    after = {c.id: c.content for c in _stored(ingested)}
    assert after == before and len(chunks) == len(before)


def test_dry_run_writes_nothing(ingested: Ingested) -> None:
    count = len(_stored(ingested))
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED, chunk_size_tokens=100, chunk_overlap_tokens=10
    )
    chunks, report = IngestionPipeline(config).run(ingested.engine, dry_run=True)
    assert not report.persisted and len(chunks) != count
    assert len(_stored(ingested)) == count and {c.strategy for c in _stored(ingested)} == {
        DEFAULT_CONFIG.strategy
    }


@pytest.mark.parametrize("strategy", [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE])
def test_every_strategy_produces_bounded_traceable_chunks(
    ingested: Ingested, strategy: ChunkingStrategy
) -> None:
    config = ChunkingConfig(strategy=strategy, chunk_size_tokens=200, chunk_overlap_tokens=30)
    with ingested.engine.connect() as connection:
        sources = list(load_sources(connection))
    chunks, report = IngestionPipeline(config).chunk_all(sources)
    assert [s.source_id for s in report.skipped] == ["DOC-0100"]  # confidential
    assert {c.document_id for c in chunks} == {s.source_id for s in sources} - {"DOC-0100"}
    assert max(c.token_count for c in chunks) <= 200
    texts = {s.source_id: source_text(s) for s in sources}
    assert all(texts[c.document_id][c.char_start : c.char_end] == c.content for c in chunks)


# --- filtering by metadata ------------------------------------------------------------------------


def _ids(ingested: Ingested, chunk_filter: ChunkFilter) -> list[str]:
    with Session(ingested.engine) as session:
        return [chunk.id for chunk in session.scalars(filtered_chunks(chunk_filter))]


def test_filter_by_service_and_source_type(ingested: Ingested) -> None:
    ids = _ids(
        ingested,
        ChunkFilter(
            services=frozenset({"payment-service"}), source_types=frozenset({SourceType.INCIDENT})
        ),
    )
    expected = sorted(
        c.id
        for c in ingested.chunks
        if c.service_id == "payment-service" and c.source_type is SourceType.INCIDENT
    )
    assert ids and sorted(ids) == expected


def test_filter_by_version_and_doc_type(ingested: Ingested) -> None:
    ids = _ids(
        ingested,
        ChunkFilter(versions=frozenset({"v2.8.1"}), services=frozenset({"payment-service"})),
    )
    types = {c.source_type for c in ingested.chunks if c.id in set(ids)}
    assert {SourceType.INCIDENT, SourceType.DEPLOYMENT, SourceType.PULL_REQUEST} <= types
    runbooks = _ids(ingested, ChunkFilter(doc_types=frozenset({DocumentType.RUNBOOK.value})))
    assert runbooks and all(
        c.source_type is SourceType.RUNBOOK for c in ingested.chunks if c.id in set(runbooks)
    )


def test_access_filter_selects_only_granted_source_and_label_pairs(ingested: Ingested) -> None:
    access = principal_for_role("t", "developer").chunk_access()
    ids = set(_ids(ingested, ChunkFilter(access=access)))
    assert ids and all(
        (c.source_type, c.access_level) in access for c in ingested.chunks if c.id in ids
    )
    hidden = [c for c in ingested.chunks if (c.source_type, c.access_level) not in access]
    assert hidden and not ids & {c.id for c in hidden}
    assert _ids(ingested, ChunkFilter(access=frozenset())) == []  # no grants: nothing
    assert not any(c.access_level is AccessLevel.CONFIDENTIAL for c in ingested.chunks)
    levels = Counter(c.access_level for c in ingested.chunks)
    assert levels[AccessLevel.MANAGER] and levels[AccessLevel.ADMIN]  # reports and policies


def test_filter_by_time_range(ingested: Ingested) -> None:
    since, until = datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)
    ids = _ids(
        ingested,
        ChunkFilter(since=since, until=until, source_types=frozenset({SourceType.INCIDENT})),
    )
    expected = sorted(
        c.id
        for c in ingested.chunks
        if c.source_type is SourceType.INCIDENT and since <= c.timestamp < until
    )
    assert ids and sorted(ids) == expected


# --- database-level provenance guarantees ---------------------------------------------------------


def _row(ingested: Ingested, **overrides: object) -> dict[str, object]:
    base = ingested.chunks[0].model_dump()
    base.update(id="PROBE#000", chunk_index=999, **overrides)
    return base


def test_database_rejects_a_chunk_without_a_source(ingested: Ingested) -> None:
    row = _row(ingested, **{k: None for k in set(SOURCE_FOREIGN_KEY.values())})
    with pytest.raises(IntegrityError), ingested.engine.begin() as connection:
        connection.execute(DocumentChunk.__table__.insert(), [row])


def test_database_rejects_a_chunk_pointing_at_another_source(ingested: Ingested) -> None:
    first = ingested.chunks[0]
    row = _row(ingested, document_id="INC-0001" if first.document_id != "INC-0001" else "INC-0002")
    with pytest.raises(IntegrityError), ingested.engine.begin() as connection:
        connection.execute(DocumentChunk.__table__.insert(), [row])


def test_deleting_a_source_deletes_its_chunks(ingested: Ingested) -> None:
    referenced = {i.runbook_id for i in ingested.dataset.incidents} | {
        i.postmortem_id for i in ingested.dataset.incidents
    }
    doc = next(d for d in ingested.dataset.documents if d.id not in referenced)
    with ingested.engine.begin() as connection:
        assert connection.execute(
            select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)
        ).all()
        connection.execute(delete(Document).where(Document.id == doc.id))
        assert (
            connection.execute(
                select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)
            ).all()
            == []
        )
        assert connection.execute(select(Incident.id).limit(1)).all()


def test_outdated_chunk_table_is_rebuilt() -> None:
    engine = sqlite_engine_with_fks()
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE document_chunks (id INTEGER PRIMARY KEY, content TEXT)")
        )
    assert ensure_chunk_table(engine) is True
    columns = {c["name"] for c in inspect(engine).get_columns("document_chunks")}
    assert {"source_type", "source_hash", "metadata", "source_incident_id"} <= columns
    assert ensure_chunk_table(engine) is False
    engine.dispose()
