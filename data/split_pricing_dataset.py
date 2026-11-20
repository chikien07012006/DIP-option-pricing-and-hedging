from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_INPUT_PATH = DATA_DIRECTORY / "pricing_dataset.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train, validation, and test CSV splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Source dataset CSV (default: data/pricing_dataset.csv).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100_000,
        help="Number of rows in the training split (default: 100000).",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=20_000,
        help="Number of rows in the validation split (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path = args.input.expanduser().resolve()

    if args.train_size <= 0 or args.validation_size <= 0:
        raise ValueError("train_size and validation_size must be positive.")

    dataset = pd.read_csv(input_path)
    required_columns = {
        "spot_price",
        "strike_price",
        "barrier_level",
        "barrier_to_spot_ratio",
        "time_to_maturity",
        "volatility",
        "risk_free_rate",
        "analytical_price",
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required columns: {sorted(missing_columns)}"
        )

    test_size = len(dataset) - args.train_size - args.validation_size
    if test_size <= 0:
        raise ValueError(
            "The dataset must contain more rows than train_size + validation_size."
        )

    shuffled = dataset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_end = args.train_size
    validation_end = train_end + args.validation_size

    train = shuffled.iloc[:train_end]
    validation = shuffled.iloc[train_end:validation_end]
    test = shuffled.iloc[validation_end:]

    output_paths = {
        "train": DATA_DIRECTORY / "train.csv",
        "validation": DATA_DIRECTORY / "validation.csv",
        "test": DATA_DIRECTORY / "test.csv",
    }
    for split_name, split_data in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        split_data.to_csv(output_paths[split_name], index=False)
        print(f"Saved {split_name}: {len(split_data):,} rows -> {output_paths[split_name]}")


if __name__ == "__main__":
    main()
