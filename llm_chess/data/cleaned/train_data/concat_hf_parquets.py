from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_BASENAME = "bestmove"
PARQUET_FILES_TO_CONCAT = [
    "bestmove_part1_5mm.parquet",
    "bestmove_part2_5mm.parquet",
    "bestmove_part3_5mm.parquet",
]
REQUIRED_COLUMNS = {"data_type", "data_subtype", "fen_board"}


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _format_sample_count_suffix(num_samples: int) -> str:
    if num_samples >= 1_000_000:
        rounded_millions = max(1, _round_half_up(num_samples / 1_000_000))
        return f"{rounded_millions}mm"

    if num_samples >= 1_000:
        rounded_thousands = max(1, _round_half_up(num_samples / 1_000))
        return f"{rounded_thousands}k"

    return str(num_samples)


def _normalize_parquet_basename(name: str) -> str:
    return name.removesuffix(".parquet")


def _parquet_paths(hf_upload_dir: Path) -> list[Path]:
    if not PARQUET_FILES_TO_CONCAT:
        raise ValueError("PARQUET_FILES_TO_CONCAT is empty.")

    parquet_paths = [hf_upload_dir / filename for filename in PARQUET_FILES_TO_CONCAT]
    missing_paths = [path.name for path in parquet_paths if not path.is_file()]
    if missing_paths:
        missing_paths_str = ", ".join(sorted(missing_paths))
        raise FileNotFoundError(
            f"Missing parquet files in {hf_upload_dir}: {missing_paths_str}"
        )

    return parquet_paths


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _normalize_object_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, bool)):
        if pd.isna(value):
            return None
        return value

    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(_json_safe_value(value), ensure_ascii=False)

    if hasattr(value, "tolist"):
        return json.dumps(value.tolist(), ensure_ascii=False)

    if pd.isna(value):
        return None

    return str(value)


def _normalize_object_columns_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    object_columns = normalized.select_dtypes(include=["object"]).columns
    for column in object_columns:
        normalized[column] = normalized[column].map(_normalize_object_value)
    return normalized


def _extract_fullmove_count(fen_board: Any) -> int | None:
    if pd.isna(fen_board):
        return None

    parts = str(fen_board).strip().split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 fields in FEN, got {len(parts)}: {fen_board}")

    try:
        return int(parts[5])
    except ValueError:
        raise ValueError(
            f"Invalid fullmove count in FEN: {fen_board}"
        ) from None


def _insert_column_after(
    frame: pd.DataFrame,
    *,
    anchor_column: str,
    new_column: str,
    values: pd.Series,
) -> pd.DataFrame:
    updated = frame.copy()
    if new_column in updated.columns:
        updated = updated.drop(columns=[new_column])

    anchor_index = updated.columns.get_loc(anchor_column)
    updated.insert(anchor_index + 1, new_column, values)
    return updated


def _drop_duplicate_fens(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_fen_mask = frame["fen_board"].duplicated(keep="first")
    duplicate_rows_dropped = int(duplicate_fen_mask.sum())
    if duplicate_rows_dropped == 0:
        return frame, 0

    deduplicated = frame.loc[~duplicate_fen_mask].reset_index(drop=True)
    return deduplicated, duplicate_rows_dropped


def add_fullmove_count_to_all_parquets(hf_upload_dir: Path | None = None) -> list[dict[str, Any]]:
    if hf_upload_dir is None:
        hf_upload_dir = Path(__file__).resolve().parent / "hf_upload"

    parquet_paths = sorted(hf_upload_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {hf_upload_dir}")

    update_stats: list[dict[str, Any]] = []

    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path).copy()
        if "fen_board" not in frame.columns:
            update_stats.append(
                {
                    "filename": parquet_path.name,
                    "rows": len(frame),
                    "updated": False,
                    "reason": "missing_fen_board",
                }
            )
            continue

        fullmove_count = frame["fen_board"].map(_extract_fullmove_count)
        updated = _insert_column_after(
            frame,
            anchor_column="fen_board",
            new_column="fullmove_count",
            values=fullmove_count,
        )
        updated.to_parquet(parquet_path, index=False)

        update_stats.append(
            {
                "filename": parquet_path.name,
                "rows": len(updated),
                "updated": True,
                "null_fullmove_count_rows": int(updated["fullmove_count"].isna().sum()),
            }
        )

    return update_stats


def main() -> None:
    hf_upload_dir = Path(__file__).resolve().parent / "hf_upload"
    parquet_paths = _parquet_paths(hf_upload_dir)

    frames: list[pd.DataFrame] = []
    input_stats: list[dict[str, str | int]] = []

    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path).copy()
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            missing_columns_str = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{parquet_path.name} is missing required columns: {missing_columns_str}"
            )

        frames.append(frame)
        input_stats.append(
            {
                "filename": parquet_path.name,
                "rows": len(frame),
                "data_types": sorted(frame["data_type"].dropna().astype(str).unique()),
                "data_subtypes": sorted(
                    frame["data_subtype"].dropna().astype(str).unique()
                ),
            }
        )

    combined = pd.concat(frames, ignore_index=True, copy=False)
    combined, duplicate_rows_dropped = _drop_duplicate_fens(combined)
    combined = _normalize_object_columns_for_parquet(combined)
    sample_count_suffix = _format_sample_count_suffix(len(combined))
    output_name = (
        f"{_normalize_parquet_basename(OUTPUT_BASENAME)}_{sample_count_suffix}.parquet"
    )
    output_path = hf_upload_dir / output_name

    combined.to_parquet(output_path, index=False)

    print(
        json.dumps(
            {
                "inputs": input_stats,
                "rows_before_dedup": sum(stat["rows"] for stat in input_stats),
                "duplicate_fen_rows_dropped": duplicate_rows_dropped,
                "rows_saved": len(combined),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
