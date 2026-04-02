from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable

from llm_chess.data.cleaned.train_data.util.fba_parser import (
    FBA_QA_TYPES,
    extract_fba_row_entries,
    summarize_fba_entries,
)
from llm_chess.data.raw.utils.board import extract_visual, visual_to_fen
from llm_chess.data.raw.utils.parsing import parse_fen
from llm_chess.prompts.chat_to_prompt import TokenizerCounter


def iter_jsonl_rows(dataset_path: str | Path) -> Iterable[dict[str, Any]]:
    dataset_path = Path(dataset_path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSONL row {line_number} in {dataset_path}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected dict rows in {dataset_path}, got {type(row).__name__}"
                )

            yield row


def iter_parquet_rows(dataset_path: str | Path) -> Iterable[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to analyze parquet datasets."
        ) from exc

    dataset_path = Path(dataset_path)
    records = pd.read_parquet(dataset_path).to_dict(orient="records")

    for row_number, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected dict rows in {dataset_path}, got {type(row).__name__} "
                f"at row {row_number}"
            )

        yield row


def extract_assistant_texts_from_chat_row(
    row: dict[str, Any],
) -> list[str]:
    chat = row.get("chat")
    if not isinstance(chat, list):
        return []

    assistant_texts: list[str] = []
    for message in chat:
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            continue

        role, content = message
        if role != "assistant":
            continue

        assistant_texts.append(content if isinstance(content, str) else str(content))

    return assistant_texts


def extract_assistant_texts_from_hf_row(
    row: dict[str, Any],
) -> list[str]:
    response = row.get("response")
    if response is None:
        return []

    return [response if isinstance(response, str) else str(response)]


def extract_fens_from_chat_row(
    row: dict[str, Any],
) -> list[str]:
    chat = row.get("chat")
    if not isinstance(chat, list):
        return []

    fens: list[str] = []
    for message in chat:
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            continue

        role, content = message
        if role != "user" or not isinstance(content, str):
            continue

        visual_board = extract_visual(content)
        fens.append(visual_to_fen(visual_board))

    return fens


def extract_fens_from_hf_row(
    row: dict[str, Any],
) -> list[str]:
    fen = row.get("fen_board")
    if not isinstance(fen, str):
        return []

    fen = fen.strip()
    return [fen] if fen else []


class MetricCounter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def update(self, row: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        pass


class AssistantTokenLengthCounter(MetricCounter):
    def __init__(
        self,
        tokenizer_version: str,
        text_extractor: Callable[[dict[str, Any]], list[str]] = (
            extract_assistant_texts_from_chat_row
        ),
        name: str = "assistant_token_lengths",
    ):
        self._token_counter = TokenizerCounter(tokenizer_version)
        self._text_extractor = text_extractor
        self._lengths: list[int] = []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        for text in self._text_extractor(row):
            self._lengths.append(self._token_counter.count(text))

    def compute(self) -> dict[str, Any]:
        if not self._lengths:
            return {
                "num_messages": 0,
                "min": None,
                "median": None,
                "mean": None,
                "max": None,
                "std_dev": None,
            }

        return {
            "num_messages": len(self._lengths),
            "min": min(self._lengths),
            "median": median(self._lengths),
            "mean": mean(self._lengths),
            "max": max(self._lengths),
            "std_dev": pstdev(self._lengths),
        }


def count_values_over_threshold(values: Iterable[int], threshold: int) -> int:
    return sum(1 for value in values if value > threshold)


class AssistantTokenThresholdCounter(MetricCounter):
    def __init__(
        self,
        tokenizer_version: str,
        threshold: int,
        text_extractor: Callable[[dict[str, Any]], list[str]] = (
            extract_assistant_texts_from_chat_row
        ),
    ):
        self._token_counter = TokenizerCounter(tokenizer_version)
        self._threshold = threshold
        self._text_extractor = text_extractor
        self._lengths: list[int] = []

    @property
    def name(self) -> str:
        return f"assistant_token_lengths_over_{self._threshold}"

    def update(self, row: dict[str, Any]) -> None:
        for text in self._text_extractor(row):
            self._lengths.append(self._token_counter.count(text))

    def compute(self) -> dict[str, Any]:
        num_messages = len(self._lengths)
        count_over_threshold = count_values_over_threshold(
            self._lengths, self._threshold
        )
        proportion_over_threshold = (
            count_over_threshold / num_messages if num_messages else None
        )

        return {
            "num_messages": num_messages,
            "threshold": self._threshold,
            "count": count_over_threshold,
            "proportion": proportion_over_threshold,
        }


FULLMOVE_HISTOGRAM_BUCKETS = (
    (0, 9, "0-9"),
    (10, 19, "10-19"),
    (20, 29, "20-29"),
    (30, 39, "30-39"),
    (40, None, "40+"),
)


def _bucket_fullmove_number(fullmove_number: int) -> str:
    for lower, upper, label in FULLMOVE_HISTOGRAM_BUCKETS:
        if upper is None and fullmove_number >= lower:
            return label
        if upper is not None and lower <= fullmove_number <= upper:
            return label
    raise ValueError(f"Unexpected fullmove number: {fullmove_number}")


class FullmoveCountBucketCounter(MetricCounter):
    def __init__(
        self,
        column: str = "fullmove_count",
        name: str = "fullmove_count_buckets",
    ) -> None:
        self._column = column
        self._name = name
        self._counts = Counter()
        self._non_null_count = 0
        self._null_count = 0
        self._values: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        value = row.get(self._column)
        if value is None:
            self._null_count += 1
            return

        try:
            fullmove_count = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid {self._column} value: {value!r}"
            ) from None

        bucket = _bucket_fullmove_number(fullmove_count)
        self._non_null_count += 1
        self._counts.update([bucket])
        self._values.append(fullmove_count)

    def compute(self) -> dict[str, Any]:
        return {
            "column": self._column,
            "num_non_null": self._non_null_count,
            "num_null": self._null_count,
            "counts": {
                label: self._counts.get(label, 0)
                for _, _, label in FULLMOVE_HISTOGRAM_BUCKETS
            },
            "ratios": {
                label: self._counts.get(label, 0) / self._non_null_count
                for _, _, label in FULLMOVE_HISTOGRAM_BUCKETS
            } if self._non_null_count else {},
            "summary": {
                "min": min(self._values) if self._values else None,
                "median": median(self._values) if self._values else None,
                "mean": mean(self._values) if self._values else None,
                "max": max(self._values) if self._values else None,
                "std_dev": pstdev(self._values) if self._values else None,
            },
        }


class BoardPromptMetadataCounter(MetricCounter):
    def __init__(
        self,
        fen_extractor: Callable[[dict[str, Any]], list[str]] = (
            extract_fens_from_chat_row
        ),
        name: str = "board_prompt_metadata",
    ) -> None:
        self._fen_extractor = fen_extractor
        self._name = name
        self._side_to_move = Counter()
        self._fullmove_histogram = Counter()
        self._fullmove_numbers: list[int] = []
        self._positions_parsed = 0

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        for fen in self._fen_extractor(row):
            fen_info = parse_fen(fen)

            side_to_move = (
                "white" if fen_info["active_color"] == "w" else "black"
            )
            fullmove_number = fen_info["fullmove_number"]
            fullmove_bucket = _bucket_fullmove_number(fullmove_number)

            self._positions_parsed += 1
            self._side_to_move.update([side_to_move])
            self._fullmove_histogram.update([fullmove_bucket])
            self._fullmove_numbers.append(fullmove_number)

    def compute(self) -> dict[str, Any]:
        if not self._positions_parsed:
            return {
                "num_positions": 0,
                "side_to_move": {},
                "fullmove_histogram": {
                    label: 0 for _, _, label in FULLMOVE_HISTOGRAM_BUCKETS
                },
                "fullmove_summary": {
                    "min": None,
                    "median": None,
                    "mean": None,
                    "max": None,
                    "std_dev": None,
                },
            }

        return {
            "num_positions": self._positions_parsed,
            "side_to_move": {
                side: self._side_to_move.get(side, 0)
                for side in ("white", "black")
            },
            "fullmove_histogram": {
                label: self._fullmove_histogram.get(label, 0)
                for _, _, label in FULLMOVE_HISTOGRAM_BUCKETS
            },
            "fullmove_summary": {
                "min": min(self._fullmove_numbers),
                "median": median(self._fullmove_numbers),
                "mean": mean(self._fullmove_numbers),
                "max": max(self._fullmove_numbers),
                "std_dev": pstdev(self._fullmove_numbers),
            },
        }


class DuplicateFenCounter(MetricCounter):
    def __init__(
        self,
        fen_extractor: Callable[[dict[str, Any]], list[str]] = (
            extract_fens_from_hf_row
        ),
        name: str = "duplicate_fen_boards",
    ) -> None:
        self._fen_extractor = fen_extractor
        self._name = name
        self._fen_counts = Counter()
        self._num_positions = 0

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        for fen in self._fen_extractor(row):
            self._num_positions += 1
            self._fen_counts.update([fen])

    def compute(self) -> dict[str, Any]:
        if not self._num_positions:
            return {
                "num_positions": 0,
                "num_unique_fens": 0,
                "num_duplicate_rows": 0,
                "proportion_duplicate_rows": None,
                "num_fens_with_duplicates": 0,
                "max_frequency": None,
            }

        duplicate_row_count = sum(
            count - 1 for count in self._fen_counts.values() if count > 1
        )
        duplicate_fen_count = sum(
            1 for count in self._fen_counts.values() if count > 1
        )

        return {
            "num_positions": self._num_positions,
            "num_unique_fens": len(self._fen_counts),
            "num_duplicate_rows": duplicate_row_count,
            "proportion_duplicate_rows": (
                duplicate_row_count / self._num_positions
            ),
            "num_fens_with_duplicates": duplicate_fen_count,
            "max_frequency": max(self._fen_counts.values()),
        }


def _normalize_categorical_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float):
        try:
            import math
        except ImportError:
            math = None

        if math is not None and math.isnan(value):
            return None

    if isinstance(value, (str, int, bool)):
        return str(value)

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)

    return str(value)


class CategoricalColumnsCounter(MetricCounter):
    def __init__(
        self,
        columns: Iterable[str],
        name: str = "categorical_columns",
    ) -> None:
        self._columns = list(columns)
        self._name = name
        self._counts_by_column = {
            column: Counter() for column in self._columns
        }
        self._non_null_counts = Counter()
        self._null_counts = Counter()

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        for column in self._columns:
            normalized_value = _normalize_categorical_value(row.get(column))
            if normalized_value is None:
                self._null_counts.update([column])
                continue

            self._non_null_counts.update([column])
            self._counts_by_column[column].update([normalized_value])

    def compute(self) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for column in self._columns:
            counts = self._counts_by_column[column]
            non_null_count = self._non_null_counts.get(column, 0)
            null_count = self._null_counts.get(column, 0)

            results[column] = {
                "num_non_null": non_null_count,
                "num_null": null_count,
                "num_unique": len(counts),
                "counts": dict(counts.most_common()),
                "ratios": {
                    key: value / non_null_count
                    for key, value in counts.most_common()
                } if non_null_count else {},
            }

        return results


class DuplicateRowSubsetCounter(MetricCounter):
    def __init__(
        self,
        columns: Iterable[str],
        name: str = "duplicate_row_subset",
    ) -> None:
        self._columns = list(columns)
        self._name = name
        self._row_counts = Counter()
        self._rows_seen = 0

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        key = tuple(_normalize_categorical_value(row.get(column)) for column in self._columns)
        self._rows_seen += 1
        self._row_counts.update([key])

    def compute(self) -> dict[str, Any]:
        if not self._rows_seen:
            return {
                "columns": self._columns,
                "rows_seen": 0,
                "num_unique_rows": 0,
                "num_duplicate_rows": 0,
                "num_duplicate_groups": 0,
                "proportion_duplicate_rows": None,
                "max_frequency": None,
            }

        duplicate_row_count = sum(
            count - 1 for count in self._row_counts.values() if count > 1
        )
        duplicate_group_count = sum(
            1 for count in self._row_counts.values() if count > 1
        )

        return {
            "columns": self._columns,
            "rows_seen": self._rows_seen,
            "num_unique_rows": len(self._row_counts),
            "num_duplicate_rows": duplicate_row_count,
            "num_duplicate_groups": duplicate_group_count,
            "proportion_duplicate_rows": duplicate_row_count / self._rows_seen,
            "max_frequency": max(self._row_counts.values()),
        }


class FBAQuestionAnswerCounter(MetricCounter):
    def __init__(
        self,
        name: str = "fba_question_answers",
    ) -> None:
        self._name = name
        self._rows_seen = 0
        self._questions_per_row: list[int] = []
        self._entries: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def update(self, row: dict[str, Any]) -> None:
        entries = extract_fba_row_entries(row)
        self._rows_seen += 1
        self._questions_per_row.append(len(entries))
        self._entries.extend(entries)

    def compute(self) -> dict[str, Any]:
        if not self._rows_seen:
            summary = summarize_fba_entries([])
            summary["rows_seen"] = 0
            summary["questions_per_row_summary"] = {
                "min": None,
                "median": None,
                "mean": None,
                "max": None,
                "std_dev": None,
            }
            return summary

        summary = summarize_fba_entries(self._entries)
        summary["rows_seen"] = self._rows_seen
        summary["questions_per_row_summary"] = {
            "min": min(self._questions_per_row),
            "median": median(self._questions_per_row),
            "mean": mean(self._questions_per_row),
            "max": max(self._questions_per_row),
            "std_dev": pstdev(self._questions_per_row),
        }
        return summary


def run_metric_counters(
    dataset_paths: Iterable[str | Path],
    counters: Iterable[MetricCounter],
    row_iterator: Callable[[str | Path], Iterable[dict[str, Any]]] = iter_jsonl_rows,
) -> dict[str, dict[str, Any]]:
    dataset_paths = list(dataset_paths)
    counters = list(counters)
    rows_processed = 0

    for dataset_path in dataset_paths:
        for row in row_iterator(dataset_path):
            rows_processed += 1
            for counter in counters:
                counter.update(row)

    results = {counter.name: counter.compute() for counter in counters}
    results["_meta"] = {
        "rows_processed": rows_processed,
        "dataset_paths": [str(Path(path)) for path in dataset_paths],
    }
    return results
