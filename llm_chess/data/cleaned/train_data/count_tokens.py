from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "count_tokens.py requires `pyarrow`. Install it before running this script."
    ) from exc

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "count_tokens.py requires `transformers`. Install it before running this script."
    ) from exc


PROMPT_TEMPLATE = """Your instructions are: {general_instruction}.

Your question is: {question}
"""

DATASET_FILES = {
    "Rejection Sampling": "rejectionsampling_combined_10k.parquet",
    "Verbalized Alpha-Beta Pruning": "vabp_values_10k.parquet",
    "Guided Synthetic": "guidedsynthetic_combined_60k.parquet",
    "Factual Board Answering": "fba_multiquestion_500k.parquet",
}


def build_prompt(general_instruction: str | None, question: str | None) -> str:
    return PROMPT_TEMPLATE.format(
        general_instruction=(general_instruction or "").strip(),
        question=(question or "").strip(),
    )


def resolve_tokenizer_source(explicit_tokenizer: str | None) -> str:
    if explicit_tokenizer:
        return explicit_tokenizer

    local_tokenizer_dir = (
        Path(__file__).resolve().parents[3] / "prompts" / "tokenizer_config" / "qwen25"
    )
    if local_tokenizer_dir.exists():
        return str(local_tokenizer_dir)

    return "Qwen/Qwen2.5-0.5B-Instruct"


def get_max_token_count(
    parquet_path: Path,
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> int:
    parquet_file = pq.ParquetFile(parquet_path)
    max_tokens = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["general_instruction", "question", "response"],
    ):
        general_instructions = batch.column(0).to_pylist()
        questions = batch.column(1).to_pylist()
        responses = batch.column(2).to_pylist()

        texts = [
            build_prompt(general_instruction, question) + (response or "")
            for general_instruction, question, response in zip(
                general_instructions,
                questions,
                responses,
            )
        ]

        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        batch_max = max(len(input_ids) for input_ids in tokenized["input_ids"])
        max_tokens = max(max_tokens, batch_max)

    return max_tokens


def format_table(rows: list[tuple[str, int]]) -> str:
    headers = ("Dataset", "Max Tokens")
    dataset_width = max(len(headers[0]), *(len(name) for name, _ in rows))
    token_width = max(len(headers[1]), *(len(str(tokens)) for _, tokens in rows))

    divider = f"+-{'-' * dataset_width}-+-{'-' * token_width}-+"
    lines = [
        divider,
        f"| {headers[0]:<{dataset_width}} | {headers[1]:>{token_width}} |",
        divider,
    ]

    for dataset_name, max_tokens in rows:
        lines.append(
            f"| {dataset_name:<{dataset_width}} | {max_tokens:>{token_width}} |"
        )

    lines.append(divider)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count the maximum prompt-plus-response token length for the parquet "
            "training datasets, excluding bestmove and bestline."
        )
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Tokenizer name or local path. Defaults to the local qwen25 tokenizer "
            "config if present, otherwise Qwen/Qwen2.5-0.5B-Instruct."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of parquet rows to tokenize per batch.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / "hf_upload"
    tokenizer_source = resolve_tokenizer_source(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    rows: list[tuple[str, int]] = []
    for dataset_name, parquet_name in DATASET_FILES.items():
        parquet_path = data_dir / parquet_name
        print(f"Processing {dataset_name}: {parquet_path.name}")
        max_tokens = get_max_token_count(parquet_path, tokenizer, args.batch_size)
        rows.append((dataset_name, max_tokens))

    print()
    print(format_table(rows))


if __name__ == "__main__":
    main()
