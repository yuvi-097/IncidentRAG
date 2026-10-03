"""pgvector HNSW indexes for ``chunk_embeddings``.

The column is dimension-less so any model fits. An HNSW index needs a fixed
dimension, so each (model, dimension) gets a *partial expression index*:

    CREATE INDEX ... ON chunk_embeddings
        USING hnsw ((embedding::vector(384)) vector_cosine_ops) WHERE model = '<model>'

Queries must use the same expression (``CAST(embedding AS vector(384))``) and a
literal ``model = '<model>'`` predicate for the planner to use it.
"""

from __future__ import annotations

import hashlib
import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

_SAFE_MODEL = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,128}$")
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


def validate_model_name(model: str) -> str:
    """Model names are embedded as SQL literals in index DDL; allow only safe characters."""
    if not _SAFE_MODEL.fullmatch(model):
        raise ValueError(f"unsupported characters in embedding model name: {model!r}")
    return model


def vector_index_name(model: str, dimension: int) -> str:
    digest = hashlib.sha256(model.encode()).hexdigest()[:10]
    return f"ix_chunk_embeddings_hnsw_{digest}_{int(dimension)}"


def ensure_vector_index(engine: Engine, model: str, dimension: int) -> str | None:
    """Create the HNSW index for ``model`` if missing. PostgreSQL only (None elsewhere)."""
    if engine.dialect.name != "postgresql":
        return None
    name = vector_index_name(validate_model_name(model), dimension)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {name} ON chunk_embeddings "
                f"USING hnsw ((embedding::vector({int(dimension)})) vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}) "
                f"WHERE model = '{model}'"
            )
        )
    return name


def refresh_statistics(engine: Engine) -> None:
    """ANALYZE ``chunk_embeddings`` after bulk writes. Without statistics the planner
    prefers the plain ``model`` btree plus a sort over the HNSW index. PostgreSQL only."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as connection:
        connection.execute(text("ANALYZE chunk_embeddings"))
