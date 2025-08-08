from llm_chess.data.raw.generation_util.process_tasks_balanced import process_tasks_balanced

# =================================
# Hyperparams
# =================================
CUR_DIR = "llm_chess/data"
MODEL_VERSION = "qwen25"
OUTPUT_FOLDER = f"{CUR_DIR}/cleaned/verl_tasks"
TASKS = [
    {"type": "predictmove", "split": "train", "samples": 8192, "data_source": f'{CUR_DIR}/raw/deepmind_data/train_20k.csv'},
    {"type": "predictmove", "split": "eval", "samples": 64, "data_source": f'{CUR_DIR}/raw/deepmind_data/evals_1k.csv'},
    {"type": "bestmove", "split": "eval", "samples": 64, "data_source": f'{CUR_DIR}/raw/deepmind_data/evals_1k.csv'},
    {"type": "worstmove", "split": "eval", "samples": 64, "data_source": f'{CUR_DIR}/raw/deepmind_data/evals_1k.csv'},
    {"type": "legalmoves", "split": "eval", "samples": 64, "data_source": f'{CUR_DIR}/raw/deepmind_data/evals_1k.csv'},
]
GENERATOR_ARGS = {
    "predictmove_min_possible_moves": 3,
    "predictmove_score_scaling": "normalize",
    "predictmove_score_cut": 0.3,
    "bestmove_provided_moves": 5,
    "bestmove_move_threshold": 0.2,
    "worstmove_provided_moves": 5,
    "worstmove_move_threshold": 0.2,
    "legalmoves_min_moves": 3
}


# =================================
# Main loop
# =================================
if __name__ == "__main__":
    process_tasks_balanced(
        tasks=TASKS,
        generator_args=GENERATOR_ARGS,
        output_folder=OUTPUT_FOLDER,
        model_version=MODEL_VERSION,
        output_type="parquet"
    )