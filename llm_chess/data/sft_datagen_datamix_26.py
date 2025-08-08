import os, json
from llm_chess.prompts.chat_to_prompt import TokenizerCounter
from llm_chess.data.raw.generation_util.sft_dataloaders import DatasetSource, load_weighted_by_samples, load_weighted_by_tokens, write_token_csv_and_stats



# Main args to adjust
MAX_SAMPLES = 750_000
MAX_TOKENS  = 7_500_000
SAMPLING_STRATEGY = "tokens"                # "samples" | "tokens" | "get_token_stats"
TOKENIZER_VERSION = "qwen25"
OUTPUT_FOLDER = "llm_chess/data/"
DATA_FOLDER = "llm_chess/data/cleaned/train_data"
DATASET_CONFIG = [
    {
        "name": "magpie",
        "files": ["magpieclean_20k.jsonl"],
        "weight": 0.3
    },
    {
        "name": "programmatic_chess_explainer",
        "files": ["combined_chessexplainer_5350.jsonl"],
        "weight": 0.1
    },
    {
        "name": "programmatic_ntp",
        "files": ["latentsft_train_contrastive_ntp_200k.jsonl", "latentsft_trainBC1mm_under_attack_200k.jsonl", "latentsft_trainBC1mm_cloze_capture_200k.jsonl", "latentsft_trainBC1mm_mobility_200k.jsonl", "latentsft_trainBC1mm_mat_adv_value_200k.jsonl", "latentsft_trainBC1mm_is_legal_200k.jsonl", "latentsft_trainBC1mm_is_check_200k.jsonl"],
        "weight": 0.1
    },
    {
        "name": "rejection_sampling_predictmove",
        "files": ["rejsampling_predictmove_balanced_5775.jsonl"],
        "weight": 0.15
    },
    {
        "name": "rejection_sampling_other",
        "files": ["rejsampling_worstmove_balanced_1724.jsonl", "rejsampling_bestmove_balanced_2535.jsonl", "rejsampling_legalmoves_balanced_604.jsonl"],
        "weight": 0.15
    },
    {
        "name": "synthetic_moves",
        "files": ["syntheticmoves_blunders_11k.jsonl", "syntheticmoves_reasonablemove_20k.jsonl"],
        "weight": 0.2
    },
    # {
    #     "name": "programmatic_bestmove",
    #     "files": ["latentsft_trainBC_bestmove_5mm_p1.jsonl", "latentsft_trainBC_bestmove_5mm_p2.jsonl", "latentsft_trainBC_bestmove_5mm_p3.jsonl"],
    #     "weight": 0.0
    # }
]



# ------------------------------ sampling ------------------------------------
sources = [
    DatasetSource(
        name=cfg["name"],
        file_paths=[f"{DATA_FOLDER}/{fname}" for fname in cfg["files"]],
        weight=cfg["weight"],
    )
    for cfg in DATASET_CONFIG
]

final_samples = None
if SAMPLING_STRATEGY == "samples":
    final_samples = load_weighted_by_samples(sources, MAX_SAMPLES)
elif SAMPLING_STRATEGY == "tokens":
    token_counter = TokenizerCounter(TOKENIZER_VERSION)
    final_samples = load_weighted_by_tokens(sources, MAX_TOKENS, token_counter)
elif SAMPLING_STRATEGY == "get_token_stats":
    csv_path = os.path.join(OUTPUT_FOLDER, "token_stats.csv")
    token_counter = TokenizerCounter(TOKENIZER_VERSION)
    write_token_csv_and_stats(sources, token_counter, csv_path)
else:
    raise ValueError("SAMPLING_STRATEGY must be 'samples' or 'tokens'")

# ------------------------------ write outputs -------------------------------
if final_samples:
    print(f"Built {len(final_samples)} examples using strategy='{SAMPLING_STRATEGY}'")
    dataset_filename = f"llamafactory_programmatic_{len(final_samples)}.json"
    with open(f"{OUTPUT_FOLDER}/{dataset_filename}", "w", encoding="utf-8") as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    datasets = {
        "llmchess_programmatic": {
            "file_name": dataset_filename,
            "columns": {"system": "system", "prompt": "user", "response": "assistant"},
        }
    }
    with open(f"{OUTPUT_FOLDER}/dataset_info.json", "w") as json_file:
        json.dump(datasets, json_file, indent=2)

    print(f"Wrote {len(final_samples)} rows → {OUTPUT_FOLDER}/{dataset_filename}")
    print(f"Dataset info saved to {OUTPUT_FOLDER}/dataset_info.json")