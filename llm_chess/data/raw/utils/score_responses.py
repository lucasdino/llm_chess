import os
import json
import pandas as pd

from .exceptions import IllegalMoveException, ParseException
from .parsing import coerce_response, extract_solution, parse_fen


PREFIX_TASK_MAP = {
    'bestmove': "choose_from_n",
    'worstmove': "choose_from_n",
    'legalmoves': "produce_list",
    'predictmove': "predict_singlemove",
    'blunder_explanations': "synthetic_generation",
    'good_move_explanations': "synthetic_generation",
}

PIECES = [
    "black pawn",
    "black rook",
    "black knight",
    "black bishop",
    "black queen",
    "black king",
    "white pawn",
    "white rook",
    "white knight",
    "white bishop",
    "white queen",
    "white king"
]



current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "fen_id_mapping.json"), "r") as f:
    FEN_ID_MAPPING = json.load(f)

# =============================================
# Dict for Response Evaluation
# =============================================
class ResponseEvaluator():
    def __init__(self, data_dir, filename):
        self.data_dir = data_dir
        self.file_prefix = filename.split("_", 1)[0]      
        assert self.file_prefix in PREFIX_TASK_MAP

        self.task_type = PREFIX_TASK_MAP[self.file_prefix]
        self.fen_id_mapping = FEN_ID_MAPPING
        self.filename = filename
        self.board_id_results = dict()
        self.results = self._instantiate_dict()
        self._compute_results()
        self._print_final_dict()

    def _compute_results(self):
        file_path = os.path.join(self.data_dir, self.filename)
        if self.filename.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as fh:
                model_responses = [json.loads(line) for line in fh]
        else:
            with open(file_path, "r", encoding="utf-8") as fh:
                model_responses = json.load(fh)

        for model_response in model_responses:
            info = model_response['info']
            prompt = model_response['prompt']
            res = model_response['model_response']
            
            # Append necessary ancillary data (fen_id, piece for legal moves, etc.)
            info['board_id'] = self.fen_id_mapping[info['board']]
            if self.task_type == "produce_list":
                for piece in PIECES:
                    if piece in prompt:
                        info['task_data'] = [piece]

            # First store metadata to track unique boards we encounter
            if info['board_id'] not in self.board_id_results:
                self.results['Count: Sample Questions'] += 1
                self.board_id_results[info['board_id']] = {
                    'score_answers': [],
                    'info': info
                }

            # Then need to get our answer / scores
            score = '<ERROR>'
            predicted_answer = '<ERROR>'
            try:
                self.results["Count: Total Generations"] += 1
                ground_truth = info['answer']

                if self.task_type == "choose_from_n":
                    answer = ground_truth['answer']
                    candidates = ground_truth['candidates']
                    predicted_answer = coerce_response(extract_solution(res), self.task_type)

                    if predicted_answer == answer:
                        score = 1
                    else:
                        if predicted_answer in candidates:
                            score = 0
                        else:
                            raise IllegalMoveException("Predicted move is not in the provided moves.")
                    self.results["Count: Legal Generations"] += 1
                    self.results["Total: Cumulative Score"] += score
                
                elif self.task_type == 'produce_list':
                    answer = ground_truth
                    predicted_answer = coerce_response(extract_solution(res), self.task_type)

                    # Compute correctness
                    num_right = 0
                    already_guessed = set()
                    for move in predicted_answer:
                        if move in answer and move not in already_guessed:
                            already_guessed.add(move)
                            num_right += 1
                    score = num_right / (len(answer) + len(predicted_answer) - num_right)
                    self.results["Count: Legal Generations"] += 1
                    self.results["Total: Cumulative Score"] += score
                    
                elif self.task_type == 'predict_singlemove':
                    answer = ground_truth
                    predicted_answer = coerce_response(extract_solution(res), self.task_type)
                    sorted_answers = sorted(answer.items(), key=lambda x: x[1])

                    if predicted_answer in answer:
                        predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                        score = predicted_move_idx/len(sorted_answers)
                    else:
                        raise IllegalMoveException("Predicted move is not in the legal moves.")
                    self.results["Count: Legal Generations"] += 1
                    self.results["Total: Cumulative Score"] += score
                    
            # Exception handling to log various errors     
            except Exception as e:
                if isinstance(e, ParseException):
                    self.results["Error: Parsing"] += 1
                elif isinstance(e, IllegalMoveException):
                    self.results["Error: Illegal Move"] += 1
                else:
                    self.results["Error: Other"] += 1
            
            # If we make it through without an error raised, can append
            self.board_id_results[info['board_id']]['score_answers'].append((score, predicted_answer))

    def _print_final_dict(self):
        """ Return finalized dict and log to wandb. """
        average_score_all = self._safe_div(self.results["Total: Cumulative Score"], self.results['Count: Total Generations'])
        average_score_legal = self._safe_div(self.results["Total: Cumulative Score"], self.results['Count: Legal Generations'])
        total_errors = self.results['Error: Illegal Move'] + self.results['Error: Parsing'] + self.results['Error: Other'] 
        error_rate = self._safe_div(total_errors, self.results['Count: Total Generations'])
        self.results['Avg. Score - All'] = average_score_all
        self.results['Avg. Score - Legal'] = average_score_legal
        self.results['Error Rate'] = error_rate
        
        print(f"{'-'*50}\nResults for {self.filename}:")
        for k, v in self.results.items():
            print(f"{k}: {v}")
        print(f"{'-'*50}\n")
    
    def _instantiate_dict(self):
        return {
            "Filename": self.filename,
            "Count: Sample Questions": 0,
            "Count: Total Generations": 0,
            "Count: Legal Generations": 0,
            "Total: Cumulative Score": 0,
            "Error: Illegal Move": 0,
            "Error: Parsing": 0,
            "Error: Other": 0
        }

    def _safe_div(self, x, y, default=0): 
        return x / y if y else default
    