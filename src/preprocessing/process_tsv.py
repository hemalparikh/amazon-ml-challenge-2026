import argparse
from pathlib import Path

import pandas as pd

from preprocessing.preprocess import preprocess_data


def process_tsv(input_path, output_path, chunksize=100_000):
    input_path = Path(input_path)
    output_path = Path(output_path)

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    ):
        processed_chunk = preprocess_data(chunk)

        processed_chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        total_rows += len(processed_chunk)
        first_chunk = False

        print(f"Processed {total_rows:,} rows")

    print(f"Completed: {input_path.name}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a TSV file in chunks."
    )

    parser.add_argument("input", help="Input TSV file")
    parser.add_argument("output", help="Output TSV file")

    args = parser.parse_args()

    process_tsv(args.input, args.output)