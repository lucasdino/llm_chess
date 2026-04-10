from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "clean_infodensity_hf.py requires `pyarrow`. Install it before running this script."
    ) from exc


PROMPT_TEMPLATE = """Your instructions are: 
{general_instruction}

Your question is: 
{question}
"""

DATASET_MAPPINGS = {
    "bestline_1p6mm.parquet": "infodensity_bl",
    "bestmove_15mm.parquet": "infodensity_bm",
    "fba_multiquestion_500k.parquet": "infodensity_fba",
    "guidedsynthetic_combined_60k.parquet": "infodensity_gs",
    "rejectionsampling_combined_10k.parquet": "infodensity_rs",
    "vabp_values_10k.parquet": "infodensity_vabp",
}

DATASET_INFO = {
    "llmchess_programmatic": {
        "file_name": None,
        "columns": {"system": "system", "prompt": "user", "response": "assistant"},
    }
}


def build_prompt(general_instruction: str | None, question: str | None) -> str:
    return PROMPT_TEMPLATE.format(
        general_instruction=(general_instruction or "").strip(),
        question=(question or "").strip(),
    )


def require_input_files(input_dir: Path, parquet_names: list[str]) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Expected input directory does not exist: {input_dir}"
        )

    missing_files = [name for name in parquet_names if not (input_dir / name).exists()]
    if missing_files:
        missing_list = "\n".join(f"  - {name}" for name in missing_files)
        raise FileNotFoundError(
            f"Missing required parquet files in {input_dir}:\n{missing_list}"
        )


def write_dataset_info(output_dir: Path, dataset_filename: str) -> None:
    dataset_info = {
        "llmchess_programmatic": {
            "file_name": dataset_filename,
            "columns": DATASET_INFO["llmchess_programmatic"]["columns"],
        }
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2)


def write_llamafactory_json(
    parquet_path: Path,
    output_dir: Path,
    batch_size: int,
) -> tuple[Path, int]:
    parquet_file = pq.ParquetFile(parquet_path)
    temp_output_path = output_dir / "llamafactory_programmatic.tmp.json"
    row_count = 0

    with temp_output_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first_record = True

        for batch_index, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["general_instruction", "question", "response"],
            ),
            start=1,
        ):
            print(
                f"  Batch {batch_index}: processing {batch.num_rows:,} rows from {parquet_path.name}"
            )
            general_instructions = batch.column(0).to_pylist()
            questions = batch.column(1).to_pylist()
            responses = batch.column(2).to_pylist()

            for general_instruction, question, response in zip(
                general_instructions,
                questions,
                responses,
            ):
                sample = {
                    "system": "",
                    "user": build_prompt(general_instruction, question),
                    "assistant": response or "",
                }
                if not first_record:
                    handle.write(",\n")
                handle.write(json.dumps(sample, ensure_ascii=False))
                first_record = False
                row_count += 1

        handle.write("]\n")

    final_output_path = output_dir / f"llamafactory_programmatic_{row_count}.json"
    temp_output_path.replace(final_output_path)
    return final_output_path, row_count


def process_dataset(
    input_dir: Path,
    parquet_name: str,
    target_folder_name: str,
    batch_size: int,
) -> None:
    parquet_path = input_dir / parquet_name
    output_dir = Path(__file__).resolve().parent / "train_data" / target_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {parquet_name} -> {output_dir}")
    output_path, row_count = write_llamafactory_json(
        parquet_path=parquet_path,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    write_dataset_info(output_dir, output_path.name)
    print(f"  Wrote {row_count:,} rows to {output_path}")
    print(f"  Wrote dataset info to {output_dir / 'dataset_info.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert parquet datasets from infodensity_raw_hf into "
            "LlamaFactory-compatible JSON datasets."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of parquet rows to process per batch.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    input_dir = data_dir / "infodensity_raw_hf"
    require_input_files(input_dir, list(DATASET_MAPPINGS.keys()))

    for parquet_name, target_folder_name in DATASET_MAPPINGS.items():
        process_dataset(
            input_dir=input_dir,
            parquet_name=parquet_name,
            target_folder_name=target_folder_name,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
