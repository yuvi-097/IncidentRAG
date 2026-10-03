"""Schema and seed against real PostgreSQL + pgvector.

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1. These tests REPLACE the contents of
the OpsRAG tables in the configured database (the same effect as
scripts/seed_db.py), so point them at a development database only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config import load_settings
from app.database.seed import seed_database, table_counts
from app.database.session import create_db_engine
from app.rag.ingestion.persistence import ensure_chunk_table
from app.synthetic.records import SyntheticDataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]


@pytest.fixture(scope="module")
def seeded(dataset: SyntheticDataset) -> Iterator[Engine]:
    settings = load_settings()
    if settings.app.environment == "production":
        pytest.skip("refusing to seed synthetic data into production")
    engine = create_db_engine(settings.database)
    ensure_chunk_table(engine)  # create_schema + upgrade an outdated chunk table
    seed_database(engine, dataset)
    yield engine
    engine.dispose()


def test_every_table_is_populated(seeded: Engine, dataset: SyntheticDataset) -> None:
    counts = table_counts(seeded)
    for table, expected in dataset.manifest.counts.items():
        assert counts[table] == expected, table


def test_embeddings_are_stored_as_pgvector(seeded: Engine) -> None:
    """Dimension-less ``vector`` column; per-model HNSW indexes are created by the embedding
    pipeline (covered in test_retrieval_postgres.py)."""
    with seeded.connect() as connection:
        column_type = connection.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'chunk_embeddings'::regclass AND attname = 'embedding'"
            )
        ).scalar_one()
    assert column_type == "vector"


def test_foreign_keys_exist(seeded: Engine) -> None:
    with seeded.connect() as connection:
        foreign_keys = connection.execute(
            text(
                "SELECT count(*) FROM information_schema.table_constraints "
                "WHERE constraint_type = 'FOREIGN KEY' AND table_schema = current_schema()"
            )
        ).scalar_one()
    assert foreign_keys >= 25


def test_multi_hop_join_from_incident_to_changed_file(
    seeded: Engine, dataset: SyntheticDataset
) -> None:
    anchor = dataset.manifest.anchors["payment-500s-v2.8.1"]
    with seeded.connect() as connection:
        rows = connection.execute(
            text(
                """
            SELECT i.id, d.version, left(pr.merge_commit_sha, 7), pr.id, cf.path, prf.patch
            FROM incidents i
            JOIN deployments d ON d.id = i.root_cause_deployment_id
            JOIN pull_requests pr ON pr.id = i.root_cause_pr_id AND pr.deployment_id = d.id
            JOIN pull_request_files prf ON prf.pull_request_id = pr.id
            JOIN code_files cf ON cf.id = prf.code_file_id
            WHERE i.id = :incident
            """
            ),
            {"incident": anchor},
        ).all()
    assert len(rows) == 1
    _, version, _, _, path, patch = rows[0]
    assert version == "v2.8.1"
    assert path == "services/payment-service/payment_service/db/database.py"
    assert "pool_size" in patch


def test_jsonb_columns_are_queryable(seeded: Engine) -> None:
    with seeded.connect() as connection:
        pool_incidents = connection.execute(
            text(
                "SELECT count(*) FROM incidents "
                "WHERE metrics ? 'max_pool_waiters' AND tags @> '[\"deployment\"]'"
            )
        ).scalar_one()
        column_type = connection.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'incidents' AND column_name = 'metrics'"
            )
        ).scalar_one()
    assert column_type == "jsonb"
    assert pool_incidents >= 1


def test_reseeding_is_idempotent(seeded: Engine, dataset: SyntheticDataset) -> None:
    before = table_counts(seeded)
    seed_database(seeded, dataset)
    after = table_counts(seeded)
    assert after == before
    with seeded.connect() as connection:
        first_log_id = connection.execute(text("SELECT min(id) FROM logs")).scalar_one()
    assert first_log_id == 1  # RESTART IDENTITY keeps log ids stable across re-seeds
