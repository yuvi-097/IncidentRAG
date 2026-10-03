"""Generate the synthetic NovaCart dataset.

    python scripts/generate_data.py [--seed 42] [--out data/generated]
                                    [--sample-dir data/sample] [--no-repo]

Deterministic: the same seed always produces byte-identical files. The output
directory is overwritten (only files this script writes there are removed).
The dataset is validated before anything is written; on validation errors the
script exits non-zero and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python scripts/generate_data.py` from the project root

from app.synthetic.generator import GenerationConfig, generate_dataset  # noqa: E402
from app.synthetic.storage import sample_dataset, write_dataset  # noqa: E402
from app.synthetic.validation import validate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=GenerationConfig.seed)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=ROOT / "data" / "sample",
        help="where to write the small committed sample ('' to skip)",
    )
    parser.add_argument(
        "--no-repo", action="store_true", help="do not materialise the code repository as files"
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    dataset = generate_dataset(GenerationConfig(seed=args.seed))
    errors = validate_dataset(dataset)
    if errors:
        print(f"dataset failed validation ({len(errors)} problems):", file=sys.stderr)
        for error in errors[:50]:
            print(f"  - {error}", file=sys.stderr)
        return 1
    write_dataset(dataset, args.out, include_repo=not args.no_repo)
    if str(args.sample_dir):
        write_dataset(sample_dataset(dataset), args.sample_dir, include_repo=False)

    elapsed = time.perf_counter() - started
    print(f"Generated dataset (seed={args.seed}) in {elapsed:.1f}s -> {args.out}")
    for table, count in dataset.manifest.counts.items():
        print(f"  {table:<22} {count:>7}")
    print("Anchor incidents:")
    for name, incident_id in dataset.manifest.anchors.items():
        print(f"  {incident_id}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
