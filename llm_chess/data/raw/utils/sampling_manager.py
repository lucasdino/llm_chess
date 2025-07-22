import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd

from .parsing import parse_fen



# ================================================================
# CONFIGS
# ================================================================
FULLMOVE_PREDMOVE_BUCKETS = [      # We will want weighting of moves to follow this distribution
    ((0, 9), 0.1),
    ((10, 19), 0.3),
    ((20, 29), 0.3),
    ((30, 39), 0.15),
    ((40, 150), 0.15)
]
FULLMOVE_BESTWORST_BUCKETS = [     # We will want weighting of moves to follow this distribution
    ((0, 9), 0.1),
    ((10, 19), 0.35),
    ((20, 29), 0.3),
    ((30, 39), 0.15),
    ((40, 150), 0.1)
]
FULLMOVE_LEGALMOVES_BUCKETS = [     # We will want weighting of moves to follow this distribution
    ((0, 9), 0.2),
    ((10, 19), 0.3),
    ((20, 29), 0.3),
    ((30, 39), 0.1),
    ((40, 150), 0.1)
]
LEGALMOVES_PIECES = [     # We want these pieces at this ratio for 'legalmoves'
    ("pawn", 0.0),
    ("knight", 0.2),
    ("bishop", 0.2),
    ("queen", 0.2),
    ("king", 0.2),
    ("rook", 0.2)
]
BESTWORST_PIECES = [      # We want these pieces at this ratio for 'bestmove / worstmove'
    ("pawn", 0.1),
    ("knight", 0.2),
    ("bishop", 0.2),
    ("queen", 0.2),
    ("king", 0.1),
    ("rook", 0.2)
]
TASK_BUCKETING_MAP = {     # This is our mapping from task to which buckets we care about
    "bestmove": [BESTWORST_PIECES, FULLMOVE_BESTWORST_BUCKETS],
    "worstmove": [BESTWORST_PIECES, FULLMOVE_BESTWORST_BUCKETS],
    "legalmoves": [LEGALMOVES_PIECES, FULLMOVE_LEGALMOVES_BUCKETS],
    "predictmove": [FULLMOVE_PREDMOVE_BUCKETS]
}



# ================================================================
# HELPERS
# ================================================================
def _bucket_fullmove(n: int,
                     bucket_def: List[Tuple[Tuple[int, int], float]]) -> str:
    """
    Map a full‑move number to its bucket label using the *task‑specific*
    bucket definition.

    Example bucket_def element: ((10, 19), 0.3) → label '10-19'
    """
    for (lo, hi), _ in bucket_def:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return ">100"          # Safety guard; shouldn't be hit with correct defs


def _desired_dict(bucket_def: List[Tuple], key_idx: int = 0) -> Dict[str, float]:
    """Convert list[(bucket, ratio)] → dict{bucket_label: ratio}."""
    return {b[key_idx] if isinstance(b[0], str)
            else f"{b[0][0]}-{b[0][1]}": b[1]
            for b in bucket_def}


def _scale_targets(ratios: Dict[str, float], capacities: Dict[str, int], total: int) -> Dict[str, int]:
    """Round desired counts to ints, then distribute leftover to buckets with highest fractional parts."""
    desired_f = {k: ratios[k] * total for k in ratios}
    floored   = {k: math.floor(v) for k, v in desired_f.items()}
    leftovers = total - sum(floored.values())

    # Rank by fractional part, grant +1 while capacity allows
    frac = sorted(((desired_f[k] - floored[k], k) for k in ratios), reverse=True)
    for _, k in frac:
        if leftovers == 0:
            break
        if floored[k] < capacities.get(k, 0):
            floored[k] += 1
            leftovers -= 1
    return floored



# ================================================================
# Main Class
# ================================================================
class SamplingManager:
    def __init__(self, iter_data: Iterable[dict], task_type: str):
        if task_type not in TASK_BUCKETING_MAP:
            raise ValueError(f"Unsupported task_type '{task_type}'. "
                             f"Expected one of {list(TASK_BUCKETING_MAP)}")

        self.task_type: str   = task_type
        self.data: List[dict] = list(iter_data)  # materialise
        self.sampled_data: Optional[List[dict]] = None

        # Unpack task‑specific bucket definitions -----------------
        mapping = TASK_BUCKETING_MAP[task_type]
        if len(mapping) == 2:                # piece buckets + full‑move buckets
            piece_def, fm_def = mapping
            self.desired_piece = _desired_dict(piece_def)
        else:                                # only full‑move buckets
            fm_def = mapping[0]
            self.desired_piece = None

        self.fm_bucket_def = fm_def
        self.desired_fm    = _desired_dict(fm_def)

    # ── Public API ───────────────────────────────────────────────────────────────
    def get_samples(self, save_path: Optional[Path] = None) -> List[dict]:
        if self.sampled_data is None:
            self._balance_distribution()
        if save_path is not None:
            len_data = len(self.sampled_data)
            save_path = save_path.with_name(f"{save_path.stem}_{len_data}.jsonl")
            save_path.write_text("\n".join(json.dumps(x)
                                           for x in self.sampled_data))
        return self.sampled_data

    # ── Internal helpers ─────────────────────────────────────────────────────────
    def _get_distribution(self) -> pd.DataFrame:
        """Return dataframe with bucketing columns and print stats."""
        rows = []
        for obj in self.data:
            info      = obj["info"]
            fullmove  = parse_fen(info["board"])["fullmove_number"]
            fm_bucket = _bucket_fullmove(fullmove, self.fm_bucket_def)

            piece = None
            if self.desired_piece is not None:
                piece_full = info["task_data"][0]   # e.g. 'white queen'
                piece      = piece_full.split()[-1].lower()

            rows.append({"obj": obj,
                         "piece": piece,
                         "fm_bucket": fm_bucket})

        df = pd.DataFrame(rows)
        total = len(df)

        # --- reporting (unchanged except uses task‑specific keys)
        print("\n── Current distribution ──")
        if self.desired_piece:
            pc_curr = (df["piece"].value_counts(normalize=True)
                       .reindex(self.desired_piece.keys(), fill_value=0))
            for p, ratio in self.desired_piece.items():
                print(f"Piece {p:<6}: current {pc_curr[p]:.2f} | "
                      f"desired {ratio:.2f}")

        fm_curr = (df["fm_bucket"].value_counts(normalize=True)
                   .reindex(self.desired_fm.keys(), fill_value=0))
        for b, ratio in self.desired_fm.items():
            print(f"Fullmove {b:<7}: current {fm_curr[b]:.2f} | "
                  f"desired {ratio:.2f}")
        return df

    def _balance_distribution(self):
        df = self._get_distribution()

        # Capacities
        cap_piece = df["piece"].value_counts().to_dict() if self.desired_piece else {}
        cap_fm    = df["fm_bucket"].value_counts().to_dict()

        # Max feasible sample count
        limits = []
        if self.desired_piece:
            limits.extend(cap_piece[p] / r for p, r in self.desired_piece.items() if r > 0)
        limits.extend(cap_fm[b] / r for b, r in self.desired_fm.items() if r > 0)
        max_total = int(math.floor(min(limits)))

        # Targets
        tgt_piece = _scale_targets(self.desired_piece, cap_piece, max_total) if self.desired_piece else {}
        tgt_fm    = _scale_targets(self.desired_fm, cap_fm,     max_total)

        # Greedy selection
        selected, used_piece, used_fm = [], Counter(), Counter()
        shuffled_idx = list(df.index)
        random.shuffle(shuffled_idx)

        for idx in shuffled_idx:
            row = df.loc[idx]
            p   = row["piece"]
            b   = row["fm_bucket"]

            # Check capacity for fullmove
            if used_fm[b] >= tgt_fm[b]:
                continue

            # Check piece capacity if applicable
            if p is not None and used_piece[p] >= tgt_piece[p]:
                continue

            # Keep sample
            selected.append(row["obj"])
            used_fm[b] += 1
            if p is not None:
                used_piece[p] += 1

            if len(selected) == max_total:
                break

        self.sampled_data = selected

        # ── Report ──
        print("\n── Final sample distribution ──")
        if self.desired_piece:
            for p in self.desired_piece:
                cur = used_piece[p] / max_total
                print(f"Piece {p:<6}: sampled {cur:.2f} | desired {self.desired_piece[p]:.2f}")
        for b in self.desired_fm:
            cur = used_fm[b] / max_total
            print(f"Fullmove {b:<7}: sampled {cur:.2f} | desired {self.desired_fm[b]:.2f}")

        thrown = len(self.data) - len(self.sampled_data)
        print(f"\nTotal kept: {len(self.sampled_data)} / {len(self.data)}  (thrown away: {thrown})")