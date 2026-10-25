"""Write an illustrative synthetic cohort to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from studysync.data import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/generated/students.csv"))
    args = parser.parse_args()

    dataset = generate_synthetic_dataset(n_samples=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset)} synthetic records to {args.output}")


if __name__ == "__main__":
    main()