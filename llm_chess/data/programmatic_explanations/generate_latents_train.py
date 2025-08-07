# generate_latents.py

from pathlib import Path
import pandas as pd

from sampling_manager import SamplingManager
from latents_generator import latents_generator

GENERATION_TYPE = "trainBC_5mm"

DATA_FILES = {
    "trainXL": "data/trainxl_1mm.csv",
    "trainBC": "data/deepmind_behavioral_cloning_train_1mm.csv",
    "trainBC_5mm": "data/deepmind_behavioral_cloning_train_5mm.csv",
    "trainBC_10mm": "data/deepmind_behavioral_cloning_train_10mm_p3.csv",
    "train": "data/train_50k.csv",
    "trainSmall": "data/train_20k.csv",
    "eval": "data/evals_1k.csv",
}

# Desired sample sizes per task
TASK_SIZES = {
    # "is_check": 200_000,
    # "large_mat_adv": 256,
    # "mat_bal": 256,
    "is_legal": 500_000,
    # "under_attack": 200_000,
    # "mat_adv_value": 200_000,
    # "win_prob": 256,
    # "mobility": 200_000,
    # "contrastive_ntp": 20_000,
    # "cloze_capture": 200_000,
    # "predict_bestmove": 5_000_000,
    # "best_move_le12": 50000,
}

TASK_TO_FN_MAP = {
    "is_check": "is-check",
    "large_mat_adv": "large-mat-adv",
    "mat_bal": "mat-bal",
    "is_legal": "is-legal",
    "under_attack": "under-attack",
    "mat_adv_value": "mat-adv-value",
    "win_prob": "win-prob",
    "mobility": "mobility",
    "contrastive_ntp": "contrastive-ntp",
    "cloze_capture": "cloze-capture",
    "predict_bestmove": "bestmove",
    "best_move_le12": "best-move-le12",
}

# Sampling criteria
BASE_CRITERIA = {
    "movecount": {
        (0, 9): 0.15,
        (10, 19): 0.3,
        (20, 29): 0.25,
        (30, 39): 0.20,
        (40, None): 0.10,
    },
    "player": {"w": 0.5, "b": 0.5},
}

GEN_CONFIG = {
    "is_check": {"tp": 0.8},
    "large_mat_adv": {"tp": 0.8},
    "mat_bal": None,
    "is_legal": {
        "choose_legal": {"legal_you": 0.5, "legal_opp": 0.1, "illegal": 0.4},
        "piece_freq": {"p": 1, "n": 2, "b": 2, "r": 2, "q": 3, "k": 1},
        "in_check": 0.1
    },
    "under_attack": {
        "legal_attack": {"attack_you": 0.4, "attack_opp": 0.2, "safe": 0.4},
        "piece_freq": {"p": 1, "n": 2, "b": 2, "r": 2, "q": 3, "k": 1},
    },
    "mat_adv_value": None,
    "win_prob": None,
    "mobility": {
        "piece_freq": {"p": 1, "n": 3, "b": 3, "r": 3, "q": 5, "k": 1},
    },
    "contrastive_ntp": {
        "min_threshold": 0.25,
        "piece_freq": {"p": 1, "n": 5, "b": 5, "r": 5, "q": 8, "k": 3},
    },
    "cloze_capture": {
        "piece_freq": {"p": 1, "n": 3, "b": 3, "r": 3, "q": 5, "k": 1},
    },
    "best_move_le8": None,
    "best_move_le9": None,
    "best_move_le10": None,
    "best_move_le11": None,
    "best_move_le12": None,
    "predict_bestmove": None,
}

TASK_CRITERIA_EXTRA = {
    "is_check": {
        "is_check": {"n": 0.50, "w": 0.25, "b": 0.25},
        "is_check_gen": {"tp": 0.40, "fp": 0.10, "tn": 0.50},
    },
    "large_mat_adv": {
        "large_mat_adv_gen": {"tp": 0.40, "fp": 0.10, "tn": 0.50},
    },
    "mat_bal": {
        "mat_bal": {"y": 0.50, "n": 0.50},
    },
    "is_legal": {
        "is_legal_gen": {"tp": 0.50, "fp": 0.10, "tn": 0.40},
        "is_legal_piece": {"p": 0.10, "b": 0.20, "n": 0.20, "r": 0.20, "q": 0.20, "k": 0.10,},
        "is_legal_in_check": {"y":0.1, "n":0.9},
    },
    "under_attack": {
        "under_attack_gen": {"tp": 0.40, "fp": 0.20, "tn": 0.40},
    },
    "mat_adv_value": {
        "mat_adv_abs": {"0-100": 0.4, "100-300": 0.3, "300+": 0.3},
    },
    "win_prob": {
        "win_prob": {"0-0.2": 0.2, "0.2-0.4": 0.2, "0.4-0.6": 0.2, "0.6-0.8": 0.2, "0.8-1": 0.2},
    },
    "mobility": {
        "mobility_piece": {"p": 0.06, "b": 0.2, "n": 0.2, "r": 0.2, "q": 0.25, "k": 0.09,},
        "mobility_moves": {"0-1": 0.25, "2-3": 0.3, "4-5": 0.3, "6+": 0.15},
    }, 
    "contrastive_ntp": {
        "contrastive_ntp": {"1": 0.5, "2": 0.5, "None": 0},
        "contrastive_ntp_piece": {"p": 0.05, "b": 0.2, "n": 0.2, "r": 0.2, "q": 0.25, "k": 0.1,},
    }, 
    "cloze_capture": {
        "cloze_piece": {"p": 0.1, "b": 0.2, "n": 0.2, "r": 0.2, "q": 0.2, "k": 0.1, "None": 0}
    },
    "best_move_le8": {},
    "best_move_le9": {},
    "best_move_le10": {},
    "best_move_le11": {},
    "best_move_le12": {},
    "predict_bestmove": {},
}

BUCKET_COLUMNS = {
    "is_check": ["movecount_bucket", "player_bucket", "is_check_bucket", "is_check_gen_bucket"],
    "large_mat_adv": ["movecount_bucket", "player_bucket", "large_mat_adv_gen_bucket"],
    "mat_bal": ["movecount_bucket", "player_bucket", "mat_bal_bucket"],
    "is_legal": ["movecount_bucket", "player_bucket", "is_legal_gen_bucket", "is_legal_piece_bucket", "is_legal_in_check_bucket"],
    "under_attack": ["movecount_bucket", "player_bucket", "under_attack_gen_bucket"],
    "mat_adv_value": ["movecount_bucket", "player_bucket", "mat_adv_abs_bucket"],
    "win_prob": ["movecount_bucket", "player_bucket", "win_prob_bucket"],
    "mobility": ["movecount_bucket", "player_bucket", "mobility_piece_bucket", "mobility_moves_bucket"],
    "contrastive_ntp": ["movecount_bucket", "player_bucket", "contrastive_ntp_piece_bucket", "contrastive_ntp_bucket"],
    "cloze_capture": ["movecount_bucket", "player_bucket", "cloze_piece_bucket"],
    "best_move_le8": ["piece_count"],
    "best_move_le9": ["piece_count"],
    "best_move_le10": ["piece_count"],
    "best_move_le11": ["piece_count"],
    "best_move_le12": ["piece_count"],
    "predict_bestmove": ["movecount_bucket", "player_bucket"],
}


def _parse_max_pieces(task: str) -> int | None:
    if task.startswith("best_move_le"):
        try:
            return int(task.split("best_move_le")[1])
        except ValueError:
            pass
    return None


def generate(task, count, base_df):
    # ---------- build chats & auxiliary columns -----------------------
    cfg = GEN_CONFIG.get(task)                 # .get() avoids KeyError
    latents_df = latents_generator(task, base_df, cfg) if cfg else latents_generator(task, base_df)

    # ---------- OPTIONAL filter for "best_move_leN" -------------------
    max_pieces = _parse_max_pieces(task)       # None for all other tasks
    if max_pieces is not None:
        print(f"Filtering for max {max_pieces} pieces in task '{task}'")
        print("NOT USING BASE_CRITERIA FOR THIS TASK")
        latents_df = latents_df[latents_df["piece_count"] <= max_pieces]

        # skip movecount / player balancing for these low-piece datasets
        if len(latents_df) < count:
            raise ValueError(f"Only {len(latents_df)} rows ≤ {max_pieces} pieces; "
                             f"can't satisfy count={count}.")
        return latents_df.sample(n=count, random_state=0).reset_index(drop=True)

    # ---------- original path for legacy tasks ------------------------
    sm   = SamplingManager(latents_df, BASE_CRITERIA)
    crit = {**BASE_CRITERIA, **TASK_CRITERIA_EXTRA.get(task, {})}
    out  = pd.DataFrame()
    while len(out) < count:
        out = pd.concat([out, sm.get_samples(count - len(out), criteria=crit)], ignore_index=True)
    return out.iloc[:count]



def print_distributions(df, cols):
    for col in cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILES[GENERATION_TYPE])
    Path("latents_train").mkdir(exist_ok=True)

    for task, count in TASK_SIZES.items():
        print(f"\n=== {task} ===")
        samples = generate(task, count, df)
        print_distributions(samples, BUCKET_COLUMNS[task])
        outpath = Path(f"latents_train/{TASK_TO_FN_MAP[task]}_{GENERATION_TYPE}-ntp_{count}.jsonl")
        samples[f"{task}_chat"].to_json(outpath, orient="records", lines=True)