"""Parsing and cleaning, including empty and malformed sources."""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.rag.chunking import ChunkingConfig, SourceFormat
from app.rag.ingestion import IngestionPipeline, RawSource, SkipSource
from app.rag.ingestion.cleaning import clean_text
from app.rag.ingestion.parsing import parse_source
from app.rag.ingestion.sources import document_source_type
from app.schemas.enums import AccessLevel, SourceType
from app.synthetic.records import SyntheticDataset


def _raw(dataset: SyntheticDataset, source_type: SourceType, source_id: str) -> RawSource:
    table = {
        SourceType.INCIDENT: dataset.incidents,
        SourceType.DEPLOYMENT: dataset.deployments,
        SourceType.CODE: dataset.code_files,
        SourceType.PULL_REQUEST: dataset.pull_requests,
    }
    records = table.get(source_type, dataset.documents)
    return RawSource(
        source_type, source_id, next(r for r in records if r.id == source_id).model_dump()
    )


# --- cleaning -------------------------------------------------------------------------------------


def test_markdown_cleaning_normalises_noise() -> None:
    raw = "\ufeffTitle\r\nline one   \r\n\x00\x07line\u200b two\t\n\n\n\n\n\nend  \n\n"
    assert clean_text(raw) == "Title\nline one\nline two\n\n\nend"


def test_markdown_cleaning_keeps_code_fences_verbatim() -> None:
    # A blank context line in a unified diff is a single space; stripping it corrupts the diff.
    raw = "# Change   \n\n```diff\n@@ -1,3 +1,3 @@\n context\n \n-old  \n+new\n```\ntext   "
    cleaned = clean_text(raw, SourceFormat.MARKDOWN)
    assert "\n \n-old  \n+new\n" in cleaned
    assert cleaned.startswith("# Change\n") and cleaned.endswith("```\ntext")


def test_code_cleaning_only_normalises_newlines_and_control_characters() -> None:
    code = "\n\ndef f():\r\n    s = 'trailing   '   \r\n\t\treturn s\x00\n\n\n\n\nx = 1\n\n"
    assert (
        clean_text(code, SourceFormat.PYTHON)
        == "def f():\n    s = 'trailing   '   \n\t\treturn s\n\n\n\n\nx = 1"
    )


@pytest.mark.parametrize("fmt", list(SourceFormat))
def test_cleaning_is_idempotent(fmt: SourceFormat) -> None:
    text = "# T  \r\n\n\n\n\nbody\u200b\n```\n keep  \n```\n  indented\n"
    assert clean_text(clean_text(text, fmt), fmt) == clean_text(text, fmt)


def test_markdown_cleaning_applies_unicode_nfc() -> None:
    assert clean_text("cafe\u0301") == "café"


# --- parsing each source type ---------------------------------------------------------------------


def test_incident_is_rendered_with_its_links(dataset: SyntheticDataset) -> None:
    anchor = dataset.manifest.anchors["payment-500s-v2.8.1"]
    parsed = parse_source(_raw(dataset, SourceType.INCIDENT, anchor))
    incident = next(i for i in dataset.incidents if i.id == anchor)
    assert parsed.title.startswith(anchor) and parsed.version == "v2.8.1"
    for fact in (
        incident.root_cause_deployment_id,
        incident.root_cause_pr_id,
        incident.runbook_id,
        "## Symptoms",
        "## Root cause",
        "## Resolution",
    ):
        assert fact in parsed.text
    assert parsed.metadata["severity"] == "SEV1" and parsed.service_id == "payment-service"


def test_rendering_does_not_depend_on_json_key_order(dataset: SyntheticDataset) -> None:
    """PostgreSQL JSONB reorders object keys; SQLite keeps insertion order."""
    raw = _raw(dataset, SourceType.INCIDENT, dataset.incidents[0].id)
    reordered = dict(reversed(list(raw.payload["metrics"].items())))
    shuffled = RawSource(raw.source_type, raw.source_id, {**raw.payload, "metrics": reordered})
    assert parse_source(shuffled).text == parse_source(raw).text


def test_pull_request_rendering_does_not_depend_on_file_row_order() -> None:
    """PostgreSQL's locale collation orders paths differently from SQLite's byte order."""
    files = [
        {
            "path": f"services/product-service/{name}",
            "additions": 1,
            "deletions": 1,
            "patch": f"--- a/{name}\n+++ b/{name}",
            "access_level": "engineering",
        }
        for name in ("product_service/catalog.py", "deploy/product-service.yaml", "README.md")
    ]
    payload = {
        "id": "PR-9000",
        "service_id": "product-service",
        "title": "t",
        "opened_at": "2026-01-01T00:00:00+00:00",
        "files": files,
    }
    forward = parse_source(RawSource(SourceType.PULL_REQUEST, "PR-9000", payload))
    backward = parse_source(
        RawSource(SourceType.PULL_REQUEST, "PR-9000", {**payload, "files": files[::-1]})
    )
    assert forward.text == backward.text


def test_deployment_lists_its_pull_requests(dataset: SyntheticDataset) -> None:
    deployment = next(
        d
        for d in dataset.deployments
        if d.version == "v2.8.1" and d.service_id == "payment-service"
    )
    raw = _raw(dataset, SourceType.DEPLOYMENT, deployment.id)
    shipped = [p.id for p in dataset.pull_requests if p.deployment_id == deployment.id]
    parsed = parse_source(
        RawSource(raw.source_type, raw.source_id, {**raw.payload, "pull_request_ids": shipped})
    )
    assert (
        all(pr in parsed.text for pr in shipped) and parsed.metadata["pull_request_ids"] == shipped
    )
    assert parsed.version == "v2.8.1" and parsed.access_level is AccessLevel.ENGINEERING


def test_pull_request_includes_diffs_and_takes_most_restrictive_file_level(
    dataset: SyntheticDataset,
) -> None:
    raw = _raw(dataset, SourceType.PULL_REQUEST, "PR-1501")
    files = [
        {
            "path": "services/payment-service/payment_service/db/database.py",
            "change_type": "modified",
            "additions": 3,
            "deletions": 2,
            "patch": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b",
            "access_level": AccessLevel.SRE,
            "service_id": "payment-service",
        }
    ]
    parsed = parse_source(
        RawSource(
            raw.source_type,
            raw.source_id,
            {**raw.payload, "files": files, "deployment_version": "v2.8.1"},
        )
    )
    assert "## Diff: services/payment-service/payment_service/db/database.py" in parsed.text
    assert "```diff" in parsed.text and parsed.access_level is AccessLevel.SRE
    assert parsed.version == "v2.8.1"


def test_documents_and_code_are_used_verbatim(dataset: SyntheticDataset) -> None:
    document = dataset.documents[0]
    parsed = parse_source(
        RawSource(document_source_type(document.doc_type), document.id, document.model_dump())
    )
    assert parsed.text == document.content and parsed.file_path == document.source_path
    code = dataset.code_files[0]
    parsed = parse_source(RawSource(SourceType.CODE, code.id, code.model_dump()))
    assert parsed.text == code.content and parsed.file_path == code.path


# --- empty and malformed sources ------------------------------------------------------------------


def _document(**overrides: Any) -> RawSource:
    payload = {
        "id": "DOC-9999",
        "doc_type": "architecture",
        "title": "T",
        "content": "# T\n\nbody",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "access_level": "engineering",
        "source_path": "docs/x.md",
    }
    payload.update(overrides)
    return RawSource(SourceType.DOCUMENTATION, "DOC-9999", payload)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (_document(content=""), "empty document content"),
        (_document(content="   \n\t "), "empty document content"),
        (_document(title=None), "missing required field(s): title"),
        (_document(updated_at="not-a-date"), "invalid updated_at"),
        (_document(access_level="top-secret"), "invalid access_level"),
        (
            RawSource(
                SourceType.INCIDENT,
                "INC-9999",
                {
                    "id": "INC-9999",
                    "service_id": "cart-service",
                    "started_at": "2026-01-01T00:00:00",
                    "access_level": "engineering",
                },
            ),
            "no symptoms, root cause or resolution",
        ),
        (
            RawSource(
                SourceType.CODE,
                "CF-9999",
                {
                    "id": "CF-9999",
                    "path": "x.py",
                    "content": "",
                    "last_modified_at": "2026-01-01T00:00:00",
                    "access_level": "engineering",
                },
            ),
            "empty file",
        ),
        (
            RawSource(
                SourceType.PULL_REQUEST,
                "PR-9999",
                {
                    "id": "PR-9999",
                    "service_id": "cart-service",
                    "title": "t",
                    "opened_at": "2026-01-01T00:00:00",
                    "files": [{"access_level": "engineering"}],
                },
            ),
            "malformed pull_request",
        ),
        (
            RawSource(SourceType.DEPLOYMENT, "DEP-9999", ["not", "a", "mapping"]),
            "payload is not a mapping",
        ),  # type: ignore[arg-type]
    ],
)
def test_malformed_sources_are_rejected_with_a_reason(raw: RawSource, reason: str) -> None:
    with pytest.raises(SkipSource, match=re.escape(reason)):
        IngestionPipeline(ChunkingConfig()).chunk_source(raw)


def test_one_bad_source_does_not_stop_the_run(dataset: SyntheticDataset) -> None:
    good = [RawSource(SourceType.INCIDENT, i.id, i.model_dump()) for i in dataset.incidents[:5]]
    sources = [*good[:2], _document(content=""), *good[2:], _document(title=None)]
    chunks, report = IngestionPipeline(ChunkingConfig()).chunk_all(sources)
    assert {c.document_id for c in chunks} == {g.source_id for g in good}
    assert [s.reason for s in report.skipped] == [
        "empty document content",
        "missing required field(s): title",
    ]
    assert report.total_sources == 7


def test_malformed_python_is_still_ingested_with_a_flag() -> None:
    raw = RawSource(
        SourceType.CODE,
        "CF-9998",
        {
            "id": "CF-9998",
            "path": "services/x/broken.py",
            "language": "python",
            "kind": "source",
            "content": "def broken(:\n    return 1\n",
            "last_modified_at": "2026-01-01T00:00:00",
            "access_level": "engineering",
        },
    )
    chunks = IngestionPipeline(ChunkingConfig()).chunk_source(raw)
    assert chunks and "SyntaxError" in chunks[0].metadata["parse_error"]


def test_markdown_with_unclosed_fence_is_ingested_with_a_flag() -> None:
    chunks = IngestionPipeline(ChunkingConfig()).chunk_source(
        _document(content="# T\n\nText\n\n```python\nprint('never closed')\n")
    )
    assert chunks[0].metadata["malformed"] == "unclosed_code_fence"
