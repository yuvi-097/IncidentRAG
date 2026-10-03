"""Embedding providers and the chunks -> batches -> embeddings -> vector store pipeline."""

from __future__ import annotations

import math
import sys
import types
from typing import Any, ClassVar

import pytest
from sqlalchemy import func, select, update

from app.config import EmbeddingSettings, RetrievalSettings
from app.database.models import ChunkEmbedding, DocumentChunk
from app.rag.chunking import ChunkingConfig
from app.rag.embeddings import (
    EmbeddingPipeline,
    EmbeddingProviderError,
    HashingEmbeddingProvider,
    build_embedding_provider,
)
from app.rag.embeddings.providers import SentenceTransformerProvider
from app.rag.embeddings.text import embedding_text
from app.rag.ingestion import IngestionPipeline
from app.synthetic.records import SyntheticDataset
from tests.retrieval.conftest import CountingProvider, Embedded, build_corpus

# --- providers ------------------------------------------------------------------------------------


def test_hashing_provider_is_deterministic_and_normalised() -> None:
    provider = HashingEmbeddingProvider(dimension=128)
    first, second = provider.embed_documents(["database pool exhausted", "database pool exhausted"])
    assert first == second and len(first) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0, rel_tol=1e-9)
    near = provider.embed_query("pool exhausted in the database")
    far = provider.embed_query("redis memory eviction")
    assert sum(a * b for a, b in zip(first, near, strict=True)) > sum(
        a * b for a, b in zip(first, far, strict=True)
    )


def test_provider_registry(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("EMBEDDING_PROVIDER", "hashing")
    clean_env.setenv("EMBEDDING_DIMENSION", "64")
    provider = build_embedding_provider(EmbeddingSettings(_env_file=None))  # type: ignore[call-arg]
    assert provider.dimension == 64 and provider.name == "hashing-v1"
    clean_env.setenv("EMBEDDING_PROVIDER", "no-such-provider")
    with pytest.raises(EmbeddingProviderError, match="unknown EMBEDDING_PROVIDER"):
        build_embedding_provider(EmbeddingSettings(_env_file=None))  # type: ignore[call-arg]


def test_model_and_prefixes_come_from_environment(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("EMBEDDING_MODEL", "intfloat/e5-small-v2")
    clean_env.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
    clean_env.setenv("EMBEDDING_DOCUMENT_PREFIX", "passage: ")
    clean_env.setenv("RETRIEVAL_TOP_K", "25")
    settings = EmbeddingSettings(_env_file=None)  # type: ignore[call-arg]
    assert (settings.model, settings.query_prefix, settings.document_prefix) == (
        "intfloat/e5-small-v2",
        "query: ",
        "passage: ",
    )
    assert RetrievalSettings(_env_file=None).top_k == 25  # type: ignore[call-arg]


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer (no model download)."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, name: str, device: str = "cpu", local_files_only: bool = False) -> None:
        if local_files_only and name == "not-cached":
            raise OSError("not in local cache")
        self.name, self.max_seq_length = name, 512
        self.tokenizer = lambda texts, add_special_tokens=True: {
            "input_ids": [t.split() for t in texts]
        }

    def get_embedding_dimension(self) -> int:
        return 768 if self.name == "wide-model" else 4

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        _FakeSentenceTransformer.calls.append({"texts": texts, **kwargs})
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeSentenceTransformer.calls = []
    return _FakeSentenceTransformer


def test_sentence_transformer_provider_applies_configured_prefixes(
    fake_sentence_transformers: Any,
) -> None:
    settings = EmbeddingSettings(
        _env_file=None,
        model="tiny",
        dimension=4,
        query_prefix="Q: ",  # type: ignore[call-arg]
        document_prefix="D: ",
        batch_size=7,
    )
    provider = SentenceTransformerProvider(settings)
    provider.embed_documents(["alpha"])
    provider.embed_query("beta")
    first, second = fake_sentence_transformers.calls
    assert first["texts"] == ["D: alpha"] and second["texts"] == ["Q: beta"]
    assert first["batch_size"] == 7 and first["normalize_embeddings"] is True
    assert provider.name == "tiny" and "document_prefix='D: '" in provider.fingerprint


def test_sentence_transformer_provider_rejects_dimension_mismatch(
    fake_sentence_transformers: Any,
) -> None:
    settings = EmbeddingSettings(_env_file=None, model="wide-model", dimension=384)  # type: ignore[call-arg]
    with pytest.raises(EmbeddingProviderError, match="768-dimensional"):
        SentenceTransformerProvider(settings)


def test_sentence_transformer_provider_downloads_only_when_not_cached(
    fake_sentence_transformers: Any,
) -> None:
    settings = EmbeddingSettings(_env_file=None, model="not-cached", dimension=4)  # type: ignore[call-arg]
    assert SentenceTransformerProvider(settings).name == "not-cached"  # fell back to a normal load


# --- the text that is embedded --------------------------------------------------------------------


def test_embedding_text_adds_a_context_header() -> None:
    text = embedding_text(
        {
            "source_type": "code",
            "title": "services/payment-service/payment_service/processor.py",
            "section": "PaymentProcessor.create_payment",
            "service_id": "payment-service",
            "content": "async def create_payment(...): ...",
        }
    )
    assert text.startswith("Code: services/payment-service/payment_service/processor.py\n")
    assert "Section: PaymentProcessor.create_payment" in text and "Service: payment-service" in text
    assert text.endswith("async def create_payment(...): ...")


def test_embedding_text_does_not_repeat_a_section_equal_to_the_title() -> None:
    text = embedding_text(
        {
            "source_type": "incident",
            "title": "INC-1: X",
            "section": "INC-1: X",
            "service_id": None,
            "content": "body",
        }
    )
    assert "Section:" not in text


# --- pipeline -------------------------------------------------------------------------------------


def test_every_chunk_is_embedded_once(embedded: Embedded) -> None:
    with embedded.engine.connect() as connection:
        rows = connection.execute(
            select(func.count())
            .select_from(ChunkEmbedding)
            .where(ChunkEmbedding.model == embedded.provider.name)
        ).scalar_one()
        dims = set(connection.execute(select(ChunkEmbedding.dimension)).scalars())
    assert embedded.report.total_chunks == len(embedded.chunks) == rows
    assert (
        embedded.report.embedded == len(embedded.chunks) and embedded.report.already_embedded == 0
    )
    assert dims == {256}


def test_batching_and_deduplication(dataset: SyntheticDataset) -> None:
    corpus = build_corpus(dataset)
    provider = CountingProvider()
    report = EmbeddingPipeline(provider, batch_size=100).run(corpus.engine)
    unique = report.embedded - report.reused_identical_text
    assert provider.embedded_texts == unique
    assert report.batches == provider.calls == math.ceil(unique / 100)
    corpus.engine.dispose()


def test_unchanged_chunks_are_never_re_embedded(dataset: SyntheticDataset) -> None:
    corpus = build_corpus(dataset)
    provider = CountingProvider()
    EmbeddingPipeline(provider).run(corpus.engine)
    first = provider.embedded_texts

    again = EmbeddingPipeline(provider).run(corpus.engine)
    assert again.embedded == 0 and provider.embedded_texts == first

    _, ingest = IngestionPipeline(ChunkingConfig()).run(corpus.engine)  # same config: no change
    assert ingest.sync["unchanged"] == len(corpus.chunks) and ingest.sync["updated"] == 0
    assert EmbeddingPipeline(provider).run(corpus.engine).embedded == 0

    with corpus.engine.begin() as connection:  # one chunk's text changes
        connection.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id == corpus.chunks[0].id)
            .values(content=corpus.chunks[0].content + " (edited)")
        )
    changed = EmbeddingPipeline(provider).run(corpus.engine)
    assert changed.embedded == 1 and provider.embedded_texts == first + 1

    forced = EmbeddingPipeline(provider).run(corpus.engine, force=True)
    assert forced.embedded == len(corpus.chunks)
    corpus.engine.dispose()


def test_rechunking_re_embeds_only_changed_chunks(dataset: SyntheticDataset) -> None:
    corpus = build_corpus(dataset)
    provider = CountingProvider()
    EmbeddingPipeline(provider).run(corpus.engine)
    _, ingest = IngestionPipeline(
        ChunkingConfig(chunk_size_tokens=200, chunk_overlap_tokens=20)
    ).run(corpus.engine)
    report = EmbeddingPipeline(provider).run(corpus.engine)
    assert report.embedded == ingest.sync["inserted"] + ingest.sync["updated"]
    assert 0 < report.embedded < report.total_chunks
    with corpus.engine.connect() as connection:
        orphans = connection.execute(
            select(func.count())
            .select_from(ChunkEmbedding)
            .where(ChunkEmbedding.chunk_id.not_in(select(DocumentChunk.id)))
        ).scalar_one()
    assert orphans == 0  # vectors of deleted/changed chunks went with them
    corpus.engine.dispose()


def test_models_are_kept_apart(dataset: SyntheticDataset) -> None:
    corpus = build_corpus(dataset)
    EmbeddingPipeline(CountingProvider(dimension=32, name="model-a")).run(corpus.engine)
    EmbeddingPipeline(CountingProvider(dimension=48, name="model-b")).run(corpus.engine)
    with corpus.engine.connect() as connection:
        per_model = dict(
            connection.execute(
                select(ChunkEmbedding.model, func.count()).group_by(ChunkEmbedding.model)
            ).all()
        )
    assert per_model == {"model-a": len(corpus.chunks), "model-b": len(corpus.chunks)}
    corpus.engine.dispose()
