from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.stage_d_power_records import materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--target-records-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selected-initialization-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        summary_path=args.summary,
        traces_dir=args.traces_dir,
        target_records_dir=args.target_records_dir,
        dataset_path=args.dataset,
        selected_initialization_sha256=(
            args.selected_initialization_sha256
        ),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
