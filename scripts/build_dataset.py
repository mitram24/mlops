"""Build the analytical dataset for the player-rating MLOps project.

Reads the Kaggle *European Soccer Database* SQLite file and produces a single, flat
``player_attributes.csv`` under ``data/01_raw`` — one row per FIFA attribute snapshot,
enriched with a handful of biometric columns (``birthday``, ``height``, ``weight``) from
the ``Player`` table so that downstream feature engineering can derive a player's age.

This script is the one place that touches the raw SQLite file; everything after it works
on the resulting CSV, exactly mirroring how the original project shipped a raw CSV in
``data/01_raw``. Re-run it to regenerate the dataset from scratch:

    python scripts/build_dataset.py --sqlite path/to/database.sqlite

A small, deterministic ``player_sample.csv`` is also written for the batch-prediction /
serving demo (the analogue of the original ``ames_sample.csv``).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "01_raw"


def build(sqlite_path: Path, sample_size: int = 25, seed: int = 42, max_rows: int | None = 50000) -> None:
    con = sqlite3.connect(str(sqlite_path))
    try:
        attrs = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
        players = pd.read_sql_query(
            "SELECT player_api_id, player_name, birthday, height, weight FROM Player", con
        )
    finally:
        con.close()

    # Enrich each attribute snapshot with the player's biometrics (for age / BMI later).
    df = attrs.merge(players, on="player_api_id", how="left")

    # The full table has ~184k snapshots. As the brief invites shipping "a sample of data
    # to run", we deterministically down-sample so the whole pipeline runs end-to-end in
    # a couple of minutes on a laptop and stays fully reproducible. Pass --max-rows 0 to
    # keep the full dataset.
    full_n = len(df)
    if max_rows and full_n > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        print(f"Down-sampled {full_n:,} -> {len(df):,} rows (seed={seed})")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "player_attributes.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {out}")

    # A reproducible sample for the serving / batch-prediction demo.
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    sample_out = RAW_DIR / "player_sample.csv"
    sample.to_csv(sample_out, index=False)
    print(f"Wrote {len(sample):,} sample rows -> {sample_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path("database.sqlite"),
        help="Path to the European Soccer Database SQLite file.",
    )
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Down-sample the analytical table to this many rows (0 = keep full dataset).",
    )
    args = parser.parse_args()
    build(args.sqlite, args.sample_size, args.seed, args.max_rows or None)
