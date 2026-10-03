"""Embedding providers.

Everything outside this module sees only ``EmbeddingProvider``. Which model is used,
and how it wants its input (instructions, prefixes, normalisation), is
configuration (``EMBEDDING_*``). Adding a provider means adding one class plus a
registry entry.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.config import EmbeddingSettings


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    name: str  # stored with every vector; vectors of different models never mix
    dimension: int
    max_input_tokens: int | None

    @property
    def fingerprint(self) -> str:
        """Everything besides the text that changes a document vector (model, prefix,
        normalisation...). Part of each embedding's ``text_hash``."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        """Model tokens per document text (to report truncation), or None if unknown."""
        ...


class SentenceTransformerProvider:
    """Any sentence-transformers model from the Hugging Face hub (or a local path)."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EmbeddingProviderError(
                "sentence-transformers is not installed (pip install -r requirements.txt)"
            ) from exc
        try:
            # Local cache first: avoids a network round-trip (and flaky retries) on every
            # start. Downloads only when the model is not cached yet.
            self._model = SentenceTransformer(
                settings.model, device=settings.device, local_files_only=True
            )
        except Exception:
            try:
                self._model = SentenceTransformer(settings.model, device=settings.device)
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"could not load embedding model {settings.model!r}: {exc}"
                ) from exc
        if settings.max_seq_length:
            self._model.max_seq_length = settings.max_seq_length
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:  # the method's name before sentence-transformers 6
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension != settings.dimension:
            raise EmbeddingProviderError(
                f"{settings.model} produces {dimension}-dimensional vectors but "
                f"EMBEDDING_DIMENSION={settings.dimension}; update the configuration"
            )
        self.name = settings.model
        self.dimension = int(dimension)
        self.max_input_tokens = int(self._model.max_seq_length)
        self._batch_size = settings.batch_size
        self._normalize = settings.normalize
        self._query_prefix = settings.query_prefix
        self._document_prefix = settings.document_prefix

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.name}|dim={self.dimension}|normalize={self._normalize}"
            f"|document_prefix={self._document_prefix!r}|max_tokens={self.max_input_tokens}"
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self._document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        # Counting only (embedding truncates on its own); verbose=False silences the
        # tokenizer's "longer than maximum sequence length" warning for over-long inputs.
        encoded = self._model.tokenizer(
            [self._document_prefix + t for t in texts], add_special_tokens=True, verbose=False
        )
        return [len(ids) for ids in encoded["input_ids"]]


class HashingEmbeddingProvider:
    """Deterministic feature-hashing vectors. No model, no download, not semantic:
    a test double and offline fallback, never a retrieval-quality baseline."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dimension: int = 384, name: str = "hashing-v1") -> None:
        self.name = name
        self.dimension = dimension
        self.max_input_tokens = None

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> HashingEmbeddingProvider:
        return cls(dimension=settings.dimension)

    @property
    def fingerprint(self) -> str:
        return f"{self.name}|dim={self.dimension}"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token, count in Counter(self._TOKEN.findall(text.lower())).items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        return None


PROVIDERS: dict[str, Callable[[EmbeddingSettings], Any]] = {
    "sentence-transformers": SentenceTransformerProvider,
    "hashing": HashingEmbeddingProvider.from_settings,
}


def build_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        supported = ", ".join(sorted(PROVIDERS))
        raise EmbeddingProviderError(
            f"unknown EMBEDDING_PROVIDER {settings.provider!r}; supported: {supported}"
        )
    return factory(settings)
