from __future__ import annotations

import json
from pathlib import Path

from llm_chess.data.cleaned.train_data.util.metric_counters import (
    AssistantTokenLengthCounter,
    AssistantTokenThresholdCounter,
    BoardPromptMetadataCounter,
    CategoricalColumnsCounter,
    DuplicateFenCounter,
    DuplicateRowSubsetCounter,
    FBAQuestionAnswerCounter,
    FullmoveCountBucketCounter,
    extract_assistant_texts_from_hf_row,
    extract_fens_from_hf_row,
    iter_parquet_rows,
    run_metric_counters,
)


TOKENIZER = "qwen25"
ASSISTANT_TOKEN_THRESHOLD = 100

JSONL_DATASET_FILES = [
    # "rejsampling_predictmove_balanced_5775.jsonl",
]
PARQUET_DATASET_FILES = [
    "bestmove_15mm.parquet",
]
METADATA_FILENAME = "bestmove_combined"

STANDARD_PARQUET_CATEGORICAL_COLUMNS = [
    "data_type",
    "data_subtype",
    "color",
    "piece_type",
]
STANDARD_PARQUET_DUPLICATE_SUBSET_COLUMNS = [
    "fen_board",
    "data_subtype",
    "color",
    "piece_type",
    # "response"
]
FBA_PARQUET_CATEGORICAL_COLUMNS = [
    "data_type",
    "data_subtype",
]
FBA_PARQUET_DUPLICATE_SUBSET_COLUMNS = [
    "fen_board",
    "individual_qa",
    "qa_type",
]


def _build_jsonl_counters() -> list:
    return [
        AssistantTokenLengthCounter(tokenizer_version=TOKENIZER),
        AssistantTokenThresholdCounter(
            tokenizer_version=TOKENIZER,
            threshold=ASSISTANT_TOKEN_THRESHOLD,
        ),
        BoardPromptMetadataCounter(),
    ]


def _build_standard_parquet_counters() -> list:
    return [
        AssistantTokenLengthCounter(
            tokenizer_version=TOKENIZER,
            text_extractor=extract_assistant_texts_from_hf_row,
        ),
        AssistantTokenThresholdCounter(
            tokenizer_version=TOKENIZER,
            threshold=ASSISTANT_TOKEN_THRESHOLD,
            text_extractor=extract_assistant_texts_from_hf_row,
        ),
        BoardPromptMetadataCounter(
            fen_extractor=extract_fens_from_hf_row,
            name="fen_board_metadata",
        ),
        FullmoveCountBucketCounter(),
        DuplicateFenCounter(fen_extractor=extract_fens_from_hf_row),
        DuplicateRowSubsetCounter(columns=STANDARD_PARQUET_DUPLICATE_SUBSET_COLUMNS),
        CategoricalColumnsCounter(columns=STANDARD_PARQUET_CATEGORICAL_COLUMNS),
    ]


def _build_fba_parquet_counters() -> list:
    return [
        AssistantTokenLengthCounter(
            tokenizer_version=TOKENIZER,
            text_extractor=extract_assistant_texts_from_hf_row,
        ),
        AssistantTokenThresholdCounter(
            tokenizer_version=TOKENIZER,
            threshold=ASSISTANT_TOKEN_THRESHOLD,
            text_extractor=extract_assistant_texts_from_hf_row,
        ),
        BoardPromptMetadataCounter(
            fen_extractor=extract_fens_from_hf_row,
            name="fen_board_metadata",
        ),
        FullmoveCountBucketCounter(),
        DuplicateFenCounter(fen_extractor=extract_fens_from_hf_row),
        DuplicateRowSubsetCounter(columns=FBA_PARQUET_DUPLICATE_SUBSET_COLUMNS),
        CategoricalColumnsCounter(columns=FBA_PARQUET_CATEGORICAL_COLUMNS),
        FBAQuestionAnswerCounter(),
    ]


def _is_fba_parquet(dataset_path: str | Path) -> bool:
    dataset_path = Path(dataset_path)
    return "fba" in dataset_path.stem.lower()


def _analyze_parquet_dataset(dataset_path: Path) -> dict:
    counters = (
        _build_fba_parquet_counters()
        if _is_fba_parquet(dataset_path)
        else _build_standard_parquet_counters()
    )
    return run_metric_counters(
        dataset_paths=[dataset_path],
        counters=counters,
        row_iterator=iter_parquet_rows,
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    jsonl_dir = base_dir / "chess"
    parquet_dir = base_dir / "hf_upload"

    results: dict[str, dict] = {}

    if JSONL_DATASET_FILES:
        jsonl_paths = [
            jsonl_dir / dataset_file for dataset_file in JSONL_DATASET_FILES
        ]
        results["jsonl"] = run_metric_counters(
            dataset_paths=jsonl_paths,
            counters=_build_jsonl_counters(),
        )

    if PARQUET_DATASET_FILES:
        parquet_paths = [
            parquet_dir / dataset_file for dataset_file in PARQUET_DATASET_FILES
        ]
        results["parquet"] = {
            parquet_path.name: _analyze_parquet_dataset(parquet_path)
            for parquet_path in parquet_paths
        }

    (parquet_dir / f"{METADATA_FILENAME}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
