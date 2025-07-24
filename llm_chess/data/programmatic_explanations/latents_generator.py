import math
import chess
import random
import pandas as pd
from typing import Dict

from prompt import _convert_fen_to_visual


# ==================================================
# Prefill Helpers / Functions
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


def _prefill_identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


# ==================================================
# Generator Helpers / Functions
# ==================================================
LATENTS_SYSPROMPT = "chess_generic.txt"
LATENTS_USERPROMPT = (
    "Here is a board in a game you're currently playing:\n{board}\n\n"
    "I want you to respond immediately with a single token -- 'Yes' or 'No' -- "
    "to answer my desired question:\n\n{question}"
)

_piece_letter_map = {            # already in file but shown for clarity
    chess.PAWN:"p", chess.KNIGHT:"n", chess.BISHOP:"b",
    chess.ROOK:"r", chess.QUEEN:"q", chess.KING:"k",
}
_letter_to_piece = {v:k for k,v in _piece_letter_map.items()}
_piece_word = {                  # for wording the question
    "p":"pawn", "n":"knight", "b":"bishop", "r":"rook", "q":"queen", "k":"king"
}


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


def _pick_legal(board: chess.Board, color: bool, weights: Dict[str, float]):
    letters = [l for l in weights if board.pieces(_letter_to_piece[l], color)]
    random.shuffle(letters)    
    seen_letters = set()
    for l in letters:
        if l in seen_letters:
            continue
        seen_letters.add(l)
        ptype = _letter_to_piece[l]
        sqs   = list(board.pieces(ptype, color))
        random.shuffle(sqs)
        for sq in sqs:
            moves = [m for m in board.legal_moves if m.from_square == sq]
            if moves:
                return l, random.choice(moves).uci()
    raise RuntimeError("Side has no legal moves.")


def _plausible_move(board: chess.Board,
                    square: chess.Square,
                    ptype: chess.PieceType,
                    color: bool) -> str:
    tmp = chess.Board(None)
    tmp.clear(); tmp.turn = color
    tmp.set_piece_at(square, chess.Piece(ptype, color))

    legal_now = {m for m in board.legal_moves}
    cand = list(tmp.generate_pseudo_legal_moves())
    random.shuffle(cand)

    # first illegal candidate
    for mv in cand:
        if mv not in legal_now and not board.is_legal(mv):
            return mv.uci()

    # otherwise try capture‑own‑piece trick with other pieces present
    for mv in cand:
        if board.color_at(mv.to_square) == color:
            return mv.uci()

    raise RuntimeError("All pseudo‑legal moves are legal in this position.")


def _generate_is_legal(df_row: pd.Series, cfg: Dict):
    """
    Robust legality‑question generator.

    cfg:
        choose_legal = {"legal_you": w1, "legal_opp": w2, "illegal": w3}
        piece_freq   = {"p": w, "n": w, …}

    Buckets:
        tp : legal move by you
        fp : legal move by opponent (illegal for you)
        tn : crafted illegal move
    """
    board         = chess.Board(df_row["FEN"])
    you, opp      = board.turn, not board.turn
    weights       = cfg["piece_freq"]
    choose_w      = cfg["choose_legal"]

    scenario = random.choices(
        ["legal_you", "legal_opp", "illegal"],
        weights=[choose_w["legal_you"], choose_w["legal_opp"], choose_w["illegal"]],
        k=1
    )[0]

    piece = move = answer = bucket = None

    # ---------------- attempt each scenario in order ------------------- #
    if scenario == "legal_you":
        try:
            piece, move = _pick_legal(board, you, weights)
            answer, bucket = "Yes", "tp"
        except RuntimeError:
            scenario = "legal_opp"     # Upon failure move to next

    if scenario == "legal_opp":
        board_opp = board.copy(); board_opp.turn = opp
        try:
            piece, move = _pick_legal(board_opp, opp, weights)
            answer, bucket = "No", "fp"
        except RuntimeError:
            scenario = "illegal"

    if scenario == "illegal":
        # try every piece type and every individual piece until an illegal move materialises
        letters = [l for l in weights if board.pieces(_letter_to_piece[l], you)]
        random.shuffle(letters)                       # avoid bias
        for piece in letters:
            ptype   = _letter_to_piece[piece]
            squares = list(board.pieces(ptype, you))
            random.shuffle(squares)
            for square in squares:
                try:
                    move = _plausible_move(board, square, ptype, you)
                    answer, bucket = "No", "tn"
                    break           # success for this scenario
                except RuntimeError:
                    continue        # this square yielded no illegal move
            else:
                continue            # try next piece type
            break                   # outer loop breaks when inner found a move
        else:
            # every attempt failed → fall back to a legal move by you
            piece, move = _pick_legal(board, you, weights)
            answer, bucket = "Yes", "tp"

    # ---------------- build output ------------------------------------- #
    question = f"Is {move} a legal move for you?"
    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=_convert_fen_to_visual(df_row["FEN"]),
                question=question)],
            ["assistant", answer],
        ]
    }
    info = {
        "is_legal_gen_bucket": bucket,
        "is_legal_piece_bucket": piece,
    }
    return chat, info


def _generate_under_attack(df_row: pd.Series, cfg: Dict):
    """
    Prompt: “Can your <piece> take their <piece>?”

    Buckets
      tp : your capture exists               (answer 'Yes')
      fp : only opponent capture exists      (answer 'No')
      tn : neither capture exists            (answer 'No')
    """
    board = chess.Board(df_row["FEN"])
    you, opp = board.turn, not board.turn
    nonking  = {chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN}

    opp_has_nonking = any(board.pieces(pt, opp) for pt in nonking)

    # --- helper: capture pair set --------------------------------------
    def capture_pairs(side, allow_king=False):
        b = board if side == board.turn else board.copy(stack=False); b.turn = side
        res = set()
        for m in b.legal_moves:
            if b.color_at(m.to_square) != (not side):
                continue
            if not allow_king and b.piece_type_at(m.to_square) == chess.KING:
                continue
            atk = _piece_letter_map[b.piece_type_at(m.from_square)]
            vic = _piece_letter_map[b.piece_type_at(m.to_square)]
            res.add((atk, vic))
        return res

    tp_set = capture_pairs(you, allow_king=not opp_has_nonking)
    opp_set = capture_pairs(opp, allow_king=True)
    fp_set = opp_set - tp_set                      # mirror capture only for opp

    # candidate piece types present on each side (exclude king victim if others)
    you_types = [l for l in cfg["piece_freq"]
                 if board.pieces(_letter_to_piece[l], you)]
    opp_types = [l for l in cfg["piece_freq"]
                 if board.pieces(_letter_to_piece[l], opp)
                    and (l != "k" or not opp_has_nonking)]

    tn_set = {(p, t) for p in you_types for t in opp_types
              if (p, t) not in tp_set and (p, t) not in fp_set}

    pools = {
        "attack_you": list(tp_set),
        "attack_opp": list(fp_set),
        "safe":       list(tn_set),
    }

    # ----- choose scenario among non‑empty pools -----------------------
    feasible = [k for k, v in pools.items() if v]
    if not feasible:                       # extremely static position
        piece, target, answer, bucket = "q", "q", "No", "tn"
    else:
        weights = [cfg["legal_attack"][k] for k in feasible]
        scenario = random.choices(feasible, weights=weights, k=1)[0]
        piece, target = random.choice(pools[scenario])
        if scenario == "attack_you":
            answer, bucket = "Yes", "tp"
        elif scenario == "attack_opp":
            answer, bucket = "No",  "fp"
        else:
            answer, bucket = "No",  "tn"

    question = f"Can your {_piece_word[piece]} take their {_piece_word[target]}?"
    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user",   LATENTS_USERPROMPT.format(
                board=_convert_fen_to_visual(df_row['FEN']), question=question)],
            ["assistant", answer],
        ]
    }
    info = {"under_attack_gen_bucket": bucket, "under_attack_piece_bucket": piece + target}
    return chat, info


# ==================================================
# Router
# ==================================================
TASK_MAP = {
    "is_check":          _generate_is_check,
    "large_mat_adv":     _generate_large_mat_adv,
    "mat_bal":           _generate_mat_bal,
    "is_legal":          _generate_is_legal,
    "under_attack":      _generate_under_attack,
}

PREFILL_TASK_MAP = {
    "is_check":          _prefill_is_check,
    "large_mat_adv":     _prefill_material_count,
    "mat_bal":           _prefill_material_count,
    "is_legal":          _prefill_identity,
    "under_attack":      _prefill_identity,
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