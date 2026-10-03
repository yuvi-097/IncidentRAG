from __future__ import annotations

from pathlib import Path

import pytest

from app.database.seed import seed_database, table_counts
from app.synthetic.records import SyntheticDataset
from app.synthetic.storage import DatasetIntegrityError, load_dataset, sample_dataset, write_dataset


@pytest.fixture(scope="module")
def written(dataset: SyntheticDataset, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("dataset")
    write_dataset(dataset, directory, include_repo=True)
    return directory


def test_round_trip_preserves_every_record(dataset: SyntheticDataset, written: Path) -> None:
    loaded = load_dataset(written)
    for table in dataset.manifest.counts:
        assert loaded.table(table) == dataset.table(table), table


def test_repository_is_materialised_as_files(dataset: SyntheticDataset, written: Path) -> None:
    code_file = next(
        f for f in dataset.code_files if f.path.endswith("payment_service/db/database.py")
    )
    on_disk = (written / "novacart-repo" / code_file.path).read_text(encoding="utf-8")
    assert on_disk == code_file.content


def test_output_is_byte_identical_across_writes(
    dataset: SyntheticDataset, written: Path, tmp_path: Path
) -> None:
    second = write_dataset(dataset, tmp_path, include_repo=False)
    first = load_dataset(written).manifest
    assert second.files == first.files


def test_tampered_file_is_rejected(dataset: SyntheticDataset, tmp_path: Path) -> None:
    write_dataset(dataset, tmp_path, include_repo=False)
    path = tmp_path / "incidents.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("SEV1", "SEV4", 1), encoding="utf-8")
    with pytest.raises(DatasetIntegrityError):
        load_dataset(tmp_path)


def test_missing_dataset_has_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"generate_data\.py"):
        load_dataset(tmp_path)


def test_sample_is_small_and_self_contained(dataset: SyntheticDataset, sqlite_engine) -> None:
    sample = sample_dataset(dataset)
    assert len(sample.incidents) >= 2 and len(sample.logs) <= 200
    # Seeding with foreign keys enforced proves every reference resolves inside the sample.
    counts = seed_database(sqlite_engine, sample)
    assert table_counts(sqlite_engine)["incidents"] == counts["incidents"]
