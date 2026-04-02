from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import chess
import pandas as pd

from llm_chess.data.cleaned.train_data.util.fba_parser import (
    extract_fba_individual_qa,
)
from llm_chess.data.cleaned.train_data.util.metric_counters import iter_jsonl_rows
from llm_chess.data.raw.utils.board import (
    extract_visual,
    get_piece_name_at_location,
    visual_to_fen,
)
from llm_chess.data.raw.utils.parsing import extract_solution
from llm_chess.prompts.chat_to_prompt import TokenizerCounter


TOKENIZER = "qwen25"
MAX_ASSISTANT_TOKENS: int | None = 100
DATASET_FILE = "latentsft_trainBC_bestmove_5mm_p3.jsonl"
PARQUET_BASENAME = "bestmove_part3"
DATA_TYPE = "Best Move"
DATA_SUBTYPE = "None"
DATA_TYPE_HANDLING: str | None = "bestmove"
REMOVE_DUPLICATE_FEN_BOARDS = False
REMOVE_DUPLICATE_RESPONSES = False


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


def _resolve_dataset_path(dataset_file: str) -> Path:
    candidate = Path(dataset_file)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    data_root = Path(__file__).resolve().parent
    preferred = data_root / "chess" / dataset_file
    fallback = data_root / dataset_file

    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not find dataset file: {dataset_file}")


def _load_prompt_mapping() -> dict[str, str]:
    mapping_path = (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "samples"
        / "prompt_mapping.json"
    )
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict):
        raise ValueError(f"Expected dict in prompt mapping, got {type(mapping).__name__}")

    return {str(key): str(value) for key, value in mapping.items()}


def _extract_chat_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    chat = row.get("chat")
    if not isinstance(chat, list):
        raise ValueError("Row is missing a valid 'chat' list.")

    sys_prompt = user_prompt = assistant_response = None

    for message in chat:
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            continue

        role, content = message
        if not isinstance(content, str):
            content = str(content)

        if role == "system" and sys_prompt is None:
            sys_prompt = content
        elif role == "user" and user_prompt is None:
            user_prompt = content
        elif role == "assistant" and assistant_response is None:
            assistant_response = content

    if sys_prompt is None or user_prompt is None or assistant_response is None:
        raise ValueError("Row is missing one of system/user/assistant messages.")

    return sys_prompt, user_prompt, assistant_response


def _map_general_instruction(sys_prompt: str, prompt_mapping: dict[str, str]) -> str:
    prompt_key = sys_prompt.removesuffix(".txt")

    if prompt_key in prompt_mapping:
        return prompt_mapping[prompt_key]

    if sys_prompt in prompt_mapping:
        return prompt_mapping[sys_prompt]

    return sys_prompt


def _extract_fen_board(question: str) -> str | None:
    try:
        visual_board = extract_visual(question)
    except ValueError:
        return None

    return visual_to_fen(visual_board)


def _extract_fullmove_count(fen_board: str | None) -> int | None:
    if fen_board is None:
        return None

    parts = fen_board.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 fields in FEN, got {len(parts)}: {fen_board}")

    try:
        return int(parts[5])
    except ValueError:
        raise ValueError(
            f"Invalid fullmove count in FEN: {fen_board}"
        ) from None


def _extract_guidedsynthetic_move(question: str) -> str | None:
    match = re.search(
        r"I'm thinking about playing\s+([a-h][1-8][a-h][1-8][qrbn]?)\.",
        question,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    return match.group(1).lower()


def _split_piece_descriptor(
    piece_descriptor: str | None,
) -> tuple[str | None, str | None]:
    if piece_descriptor is None:
        return None, None

    parts = piece_descriptor.split(" ", maxsplit=1)
    if len(parts) != 2:
        return None, None

    return parts[0], parts[1]


def _extract_last_answer_text(response: str) -> str | None:
    try:
        return extract_solution(response)
    except Exception:
        return None


def _extract_singlemove_generated_answer(response: str) -> str | None:
    answer_text = _extract_last_answer_text(response)
    if answer_text is None:
        return None

    match = re.search(
        r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b",
        answer_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    return match.group(1).lower()


def _extract_legalmoves_generated_answer(response: str) -> list[str] | None:
    answer_text = _extract_last_answer_text(response)
    if answer_text is None:
        return None

    try:
        parsed = ast.literal_eval(answer_text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple)):
        moves = []
        for item in parsed:
            if not isinstance(item, str):
                continue

            match = re.fullmatch(
                r"[a-h][1-8][a-h][1-8][qrbn]?",
                item.strip().lower(),
            )
            if match is not None:
                moves.append(match.group(0))

        return moves or None

    moves = re.findall(
        r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b",
        answer_text,
        flags=re.IGNORECASE,
    )
    return [move.lower() for move in moves] or None


def _extract_bestline_first_move(response: str) -> str | None:
    answer_text = _extract_last_answer_text(response) or response
    tokens = answer_text.strip().split()
    if not tokens:
        return None

    first_token = tokens[0].lower()
    if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", first_token) is None:
        return None

    return first_token


def _extract_bestline_value(response: str) -> str | None:
    answer_text = _extract_last_answer_text(response) or response
    tokens = answer_text.strip().split()
    if not tokens:
        return None

    last_token = tokens[-1]
    if last_token.lower() == "mate":
        return "mate"

    match = re.fullmatch(r"\[[\u0394\u03b4]([+-]?\d+)\]", last_token)
    if match is None:
        return None

    return match.group(1)


def _extract_bestmove_generated_answer(response: str) -> str | None:
    answer_text = _extract_last_answer_text(response) or response
    move = re.sub(r"\s+", "", answer_text)
    return move or None


def _extract_piece_info_from_move(
    fen_board: str | None,
    move: str | None,
) -> tuple[str | None, str | None]:
    if fen_board is None or move is None:
        return None, None

    piece_descriptor = get_piece_name_at_location(
        fen_board,
        move[:2],
    )
    return _split_piece_descriptor(piece_descriptor)


def _extract_piece_info_from_san_move(
    fen_board: str | None,
    san_move: str | None,
) -> tuple[str | None, str | None]:
    if fen_board is None or san_move is None:
        return None, None

    try:
        board = chess.Board(fen_board)
        move = board.parse_san(san_move)
    except ValueError:
        return None, None

    piece = board.piece_at(move.from_square)
    if piece is None:
        return None, None

    color = "white" if piece.color == chess.WHITE else "black"
    piece_type = chess.piece_name(piece.piece_type)
    return color, piece_type


def _extract_piece_info_from_legalmoves_question(
    question: str,
) -> tuple[str | None, str | None]:
    match = re.search(
        r"legal moves for the\s+(white|black)\s+([a-z]+)\s+at\s+[a-h][1-8]",
        question,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None

    return match.group(1).lower(), match.group(2).lower()


def _build_type_specific_columns(
    *,
    fen_board: str | None,
    question: str,
    response: str,
) -> dict[str, Any]:
    if DATA_TYPE_HANDLING == "fba":
        individual_qa, qa_type = extract_fba_individual_qa(
            question_prompt=question,
            response=response,
        )
        return {
            "individual_qa": individual_qa,
            "qa_type": qa_type,
        }

    if DATA_TYPE_HANDLING == "guidedsynthetic":
        candidate_move = _extract_guidedsynthetic_move(question)
        color, piece_type = _extract_piece_info_from_move(
            fen_board=fen_board,
            move=candidate_move,
        )
        return {
            "candidate_move": candidate_move,
            "color": color,
            "piece_type": piece_type,
        }

    if DATA_TYPE_HANDLING == "rejectionsampling_singlemove":
        generated_answer = _extract_singlemove_generated_answer(response)
        color, piece_type = _extract_piece_info_from_move(
            fen_board=fen_board,
            move=generated_answer,
        )
        return {
            "generated_answer": generated_answer,
            "color": color,
            "piece_type": piece_type,
        }

    if DATA_TYPE_HANDLING == "bestline":
        first_move = _extract_bestline_first_move(response)
        color, piece_type = _extract_piece_info_from_move(
            fen_board=fen_board,
            move=first_move,
        )
        value = _extract_bestline_value(response)
        return {
            "first_move": first_move,
            "value": value,
            "color": color,
            "piece_type": piece_type,
        }

    if DATA_TYPE_HANDLING == "bestmove":
        generated_answer = _extract_bestmove_generated_answer(response)
        color, piece_type = _extract_piece_info_from_san_move(
            fen_board=fen_board,
            san_move=generated_answer,
        )
        return {
            "generated_answer": generated_answer,
            "color": color,
            "piece_type": piece_type,
        }

    if DATA_TYPE_HANDLING == "vabp":
        generated_answer = _extract_singlemove_generated_answer(response)
        color, piece_type = _extract_piece_info_from_move(
            fen_board=fen_board,
            move=generated_answer,
        )
        return {
            "generated_answer": generated_answer,
            "color": color,
            "piece_type": piece_type,
        }

    if DATA_TYPE_HANDLING == "rejectionsampling_legalmoves":
        generated_answer = _extract_legalmoves_generated_answer(response)
        color, piece_type = _extract_piece_info_from_legalmoves_question(
            question
        )
        return {
            "generated_answer": generated_answer,
            "color": color,
            "piece_type": piece_type,
        }

    return {}


def _output_columns() -> list[str]:
    columns = [
        "general_instruction",
        "question",
        "response",
        "fen_board",
        "fullmove_count",
        "data_type",
        "data_subtype",
    ]

    if DATA_TYPE_HANDLING == "fba":
        columns.extend(["individual_qa", "qa_type"])
    elif DATA_TYPE_HANDLING == "guidedsynthetic":
        columns.extend(["candidate_move", "color", "piece_type"])
    elif DATA_TYPE_HANDLING == "bestline":
        columns.extend(["first_move", "value", "color", "piece_type"])
    elif DATA_TYPE_HANDLING in {
        "bestmove",
        "vabp",
        "rejectionsampling_singlemove",
        "rejectionsampling_legalmoves",
    }:
        columns.extend(["generated_answer", "color", "piece_type"])

    return columns


def _build_export_rows(
    dataset_path: Path,
    prompt_mapping: dict[str, str],
    tokenizer: TokenizerCounter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_rows: list[dict[str, Any]] = []
    seen_fen_boards: set[str] = set()
    total_rows = 0
    skipped_for_length = 0
    skipped_for_missing_fen = 0
    skipped_for_duplicate_fen = 0

    for row in iter_jsonl_rows(dataset_path):
        total_rows += 1
        sys_prompt, user_prompt, assistant_response = _extract_chat_fields(row)
        assistant_tokens = tokenizer.count(assistant_response)

        if (
            MAX_ASSISTANT_TOKENS is not None
            and assistant_tokens > MAX_ASSISTANT_TOKENS
        ):
            skipped_for_length += 1
            continue

        fen_board = _extract_fen_board(user_prompt)
        if fen_board is None:
            skipped_for_missing_fen += 1
        else:
            if (
                REMOVE_DUPLICATE_FEN_BOARDS
                and fen_board in seen_fen_boards
            ):
                skipped_for_duplicate_fen += 1
                continue
            seen_fen_boards.add(fen_board)

        export_row: dict[str, Any] = {
            "general_instruction": _map_general_instruction(
                sys_prompt=sys_prompt,
                prompt_mapping=prompt_mapping,
            ),
            "question": user_prompt,
            "response": assistant_response,
            "fen_board": fen_board,
            "fullmove_count": _extract_fullmove_count(fen_board),
            "data_type": DATA_TYPE,
            "data_subtype": DATA_SUBTYPE,
        }
        export_row.update(
            _build_type_specific_columns(
                fen_board=fen_board,
                question=user_prompt,
                response=assistant_response,
            )
        )

        export_rows.append(export_row)

    stats = {
        "dataset_path": str(dataset_path),
        "tokenizer": TOKENIZER,
        "max_assistant_tokens": MAX_ASSISTANT_TOKENS,
        "data_type": DATA_TYPE,
        "data_subtype": DATA_SUBTYPE,
        "data_type_handling": DATA_TYPE_HANDLING,
        "remove_duplicate_fen_boards": REMOVE_DUPLICATE_FEN_BOARDS,
        "remove_duplicate_responses": REMOVE_DUPLICATE_RESPONSES,
        "rows_seen": total_rows,
        "rows_saved": len(export_rows),
        "rows_skipped_for_length": skipped_for_length,
        "rows_with_missing_fen_board": skipped_for_missing_fen,
        "rows_skipped_for_duplicate_fen_board": skipped_for_duplicate_fen,
        "unique_fen_boards_saved": len(seen_fen_boards),
    }
    return export_rows, stats


def _response_dedup_subset() -> list[str] | None:
    if not REMOVE_DUPLICATE_RESPONSES:
        return None

    if DATA_TYPE_HANDLING == "fba":
        return None

    return ["response"]


def main() -> None:
    dataset_path = _resolve_dataset_path(DATASET_FILE)
    prompt_mapping = _load_prompt_mapping()
    tokenizer = TokenizerCounter(TOKENIZER)

    export_rows, stats = _build_export_rows(
        dataset_path=dataset_path,
        prompt_mapping=prompt_mapping,
        tokenizer=tokenizer,
    )

    export_frame = pd.DataFrame(
        export_rows,
        columns=_output_columns(),
    )
    rows_before_response_dedup = len(export_frame)
    response_dedup_subset = _response_dedup_subset()
    if response_dedup_subset is not None:
        export_frame = export_frame.drop_duplicates(
            subset=response_dedup_subset
        ).reset_index(drop=True)

    rows_removed_for_duplicate_response = (
        rows_before_response_dedup - len(export_frame)
    )

    sample_count_suffix = _format_sample_count_suffix(len(export_frame))
    parquet_name = f"{_normalize_parquet_basename(PARQUET_BASENAME)}_{sample_count_suffix}.parquet"
    output_dir = Path(__file__).resolve().parent / "hf_upload"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / parquet_name

    export_frame.to_parquet(output_path, index=False)

    stats["rows_saved_before_duplicate_response_removal"] = rows_before_response_dedup
    stats["rows_removed_for_duplicate_response"] = (
        rows_removed_for_duplicate_response
    )
    stats["response_dedup_subset"] = response_dedup_subset
    stats["rows_saved"] = len(export_frame)
    stats["output_path"] = str(output_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
