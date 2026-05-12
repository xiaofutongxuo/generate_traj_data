#!/usr/bin/env python3
"""Quick viewer to inspect generated trajectory parquet files."""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Find all parquet files
    parquet_files = list(output_dir.rglob("*.egomotion.parquet"))
    print(f"Found {len(parquet_files)} parquet files\n")

    for pf in parquet_files:
        print(f"{'='*60}")
        print(f"File: {pf.relative_to(output_dir)}")
        print(f"{'='*60}")

        df = pd.read_parquet(pf)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}\n")

        # Show trajectory statistics
        for i, row in df.iterrows():
            x, y = row['x'], row['y']
            print(f"  Traj {i}:")
            print(f"    Start: ({x[0]:.2f}, {y[0]:.2f})")
            print(f"    End:   ({x[-1]:.2f}, {y[-1]:.2f})")
            print(f"    Length: {len(x)} steps")
            print(f"    Avg speed: {sum(row['vx'])/len(row['vx']):.2f} m/s")

        print()


if __name__ == "__main__":
    main()
