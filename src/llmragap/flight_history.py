from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLIGHTS_PARQUET = PROJECT_ROOT / "example_data" / "us_flights_2024_all.parquet"


def parquet_to_csv(parquet_path: Path = FLIGHTS_PARQUET, csv_path: Path | None = None) -> Path:
    """Read every row and column of a parquet file and save it as CSV next to it."""
    csv_path = csv_path or Path(parquet_path).with_suffix(".csv")
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df):,} rows x {len(df.columns)} columns to {csv_path}")
    return csv_path


if __name__ == "__main__":
    parquet_to_csv()
