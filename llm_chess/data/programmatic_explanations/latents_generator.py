import math
import chess
import random
import pandas as pd
from typing import Dict

from prompt import _convert_fen_to_visual


# ==================================================
# Prefill Helpers
# ==================================================
_PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}

def _material_cp(board: chess.Board) -> int:
    """Static material score in centipawns (white‑positive, black‑negative)."""
    score = 0
    for ptype, val in _PIECE_VALUE.items():
        score += val * (len(board.pieces(ptype, chess.WHITE)) -
                        len(board.pieces(ptype, chess.BLACK)))
    return score


def _prefill_is_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'is_check_bucket' column:
        n : no side in check
        w : white king is in check
        b : black king is in check
    Returns a *copy* of the input DataFrame with the new column.
    """
    def status(fen: str) -> str:
        board = chess.Board(fen)
        if not board.is_check():
            return "n"
        return "w" if board.turn else "b"   # board.turn == True  -> white to move

    df = df.copy()
    df["is_check_bucket"] = df["FEN"].apply(status)
    return df


def _prefill_material_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      • 'mat_adv'              : centipawn material score (white positive)
      • 'large_mat_adv_bucket' : 'y' if |mat_adv| > 300 else 'n'
      • 'mat_bal_bucket' : 'y' if |mat_adv| < 120 else 'n'
    """
    def cp(fen): return _material_cp(chess.Board(fen))
    df = df.copy()
    df["mat_adv"] = df["FEN"].apply(cp)
    df["large_mat_adv_bucket"] = df["mat_adv"].abs().gt(300).map({True: "y", False: "n"})
    df["mat_bal_bucket"] = df["mat_adv"].abs().lt(120).map({True: "y", False: "n"})
    return df



# ==================================================
# Helpers
# ==================================================
LATENTS_SYSPROMPT = "chess_generic.txt"
LATENTS_USERPROMPT = (
    "Here is a board in a game you're currnetly playing:\n{board}\n\n"
    "I want you to respond immediately with a single token -- 'Yes' or 'No' -- "
    "to answer my desired question:\n\n{question}"
)


def _generate_is_check(df_row: pd.Series, config_args: Dict):
    # Cases to generate our data
    if df_row["is_check_bucket"] == "n":
        question = (
            "Is the black king in check?"
            if random.random() < 0.5
            else "Is the white king in check?"
        )
        answer = "No"
        cat = "tn"
    else:
        if random.random() < config_args["tp"]:
            answer = "Yes"
            cat = "tp"
        else:
            answer = "No"
            cat = "fp"

        play_black = (
            (answer == "Yes" and df_row["is_check_bucket"] == "b")
            or (answer == "No" and df_row["is_check_bucket"] == "w")
        )
        question = (
            "Is the black king in check?" if play_black else "Is the white king in check?"
        )

    # Formatting outputs
    sys_prompt = LATENTS_SYSPROMPT
    user_prompt = LATENTS_USERPROMPT.format(
        board=_convert_fen_to_visual(df_row["FEN"]), question=question
    )
    df_info = {"is_check_gen_bucket": cat}
    chat = {"chat": [["system", sys_prompt], ["user", user_prompt], ["assistant", answer]]}
    return chat, df_info


def _generate_large_mat_adv(df_row: pd.Series, config_args: Dict):
    if df_row["large_mat_adv_bucket"] == "n":
        ask_white = random.random() < 0.5
        question  = f"Does {'white' if ask_white else 'black'} have a material advantage?"
        answer, cat = "No", "tn"
    else:
        advantaged = "white" if df_row["mat_adv"] > 0 else "black"
        disadvantaged = "black" if advantaged == "white" else "white"

        if random.random() < config_args["tp"]:
            question = f"Does {advantaged} have a material advantage?"
            answer, cat = "Yes", "tp"
        else:
            question = f"Does {disadvantaged} have a material advantage?"
            answer, cat = "No", "fp"

    sys_prompt = LATENTS_SYSPROMPT
    user_prompt = LATENTS_USERPROMPT.format(
        board=_convert_fen_to_visual(df_row["FEN"]),
        question=question,
    )
    df_info = {"large_mat_adv_gen_bucket": cat}
    chat = {"chat": [["system", sys_prompt], ["user", user_prompt], ["assistant", answer]]}
    return chat, df_info


def _generate_mat_bal(df_row: pd.Series, config_args: Dict):
    """
    Q: “Is the game materially balanced?”
       (balance = |mat_adv| < 120)
    """
    question = "Is the game materially balanced?"
    answer = "Yes" if df_row["mat_bal_bucket"] == "y" else "No"

    sys_prompt  = LATENTS_SYSPROMPT
    user_prompt = LATENTS_USERPROMPT.format(
        board=_convert_fen_to_visual(df_row["FEN"]),
        question=question,
    )
    chat    = {"chat": [["system", sys_prompt],
                        ["user",   user_prompt],
                        ["assistant", answer]]}
    return chat, None


# ==================================================
# Router
# ==================================================
TASK_MAP = {
    "is_check":          _generate_is_check,
    "large_mat_adv":     _generate_large_mat_adv,
    "mat_bal":           _generate_mat_bal,
}

PREFILL_TASK_MAP = {
    "is_check":          _prefill_is_check,
    "large_mat_adv":     _prefill_material_count,
    "mat_bal":           _prefill_material_count,
}


# ==================================================
# Main Prompt Generator
# ==================================================
def latents_generator(task: str, df: pd.DataFrame, config_args: Dict = None) -> pd.DataFrame:
    """
    Adds `{task}_chat` plus any columns returned in the `df_info` dict
    (e.g. 'is_check_gen_cat') for every row in *df*.
    Prefill functions in PREFILL_TASK_MAP run automatically.
    """
    if task in PREFILL_TASK_MAP:
        df = PREFILL_TASK_MAP[task](df)

    chat_col = f"{task}_chat"
    chat_data_list = []
    info_buffers: Dict[str, list] = {}        # {info_key: [vals…]}

    for _, row in df.iterrows():
        chat_data, df_info = TASK_MAP[task](row, config_args)
        chat_data_list.append(chat_data)

        if df_info:
            for k, v in df_info.items():
                info_buffers.setdefault(k, []).append(v)

    # attach new columns
    df[chat_col] = chat_data_list
    for k, vals in info_buffers.items():
        df[k] = vals

    return df