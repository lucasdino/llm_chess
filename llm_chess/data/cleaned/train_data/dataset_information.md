# Chess Reasoning Data -- Schemas and Distributions   
*TLDR; We took great care to ensure our data was balanced across several axes -- you'll see these below in each dataset section.*

## Dataset Overviews   
We recommend referring to our paper [LINK TBU] for more information on each dataset and samples. However, below is a quick overview of each data type included below:
| Dataset | Num Samples | Num Tokens | Summary |
| --- | ---: | ---: | --- |
| `Rejection Sampling` | 10,483 | 7,448,006 | Responses from Llama 4 Maverick that were correct / exceeding a threshold across four evaluation tasks. |
| `Verbalized Alpha-Beta Pruning` | 10,000 | 4,173,183 | Programmatically generated data that emulates alpha-beta pruning using language (from phrase banks). |
| `Guided Synthetic` | 60,071 | 7,590,338 | Synthetic data (largely from gpt-oss-120b) that gives a move verdict where the *teacher* (generator) is provided chess engine information. |
| `Factual Board Answering` | 500,000 | 4,175,854 | Programmatically generated. Given a board, 4-6 questions are asked that primarily require single-token responses about board state. |
| `Best Move` | 14,848,802 | 59,424,394 | Given a board, assistant is asked to provide the top move (per Stockfish). |
| `Best Line` | 1,604,684 | 41,175,357 | Given a board, assistant is asked to predict the likely line of 4-6 plies (i.e., half-moves) per Stockfish. |

*Note: Token counts per Qwen2.5 tokenizer.* 

## Shared Columns   
All exported parquet datasets include these base columns. We specify any task-specific additional columns in each bucket below.

| Column | Meaning |
| --- | --- |
| `general_instruction` | Resolved system prompt text from `prompt_mapping.json`. |
| `question` | User prompt shown to the model. |
| `response` | Assistant response from the source dataset. |
| `fen_board` | Parsed FEN board state when available. |
| `fullmove_count` | Fullmove number parsed from the final field of `fen_board`. |
| `data_type` | High-level dataset family, such as `Guided Synthetic` or `Rejection Sampling`. |
| `data_subtype` | Task subtype within the family, such as `Good Moves`, `Best Move`, or `Legal Moves`. |


# Rejection Sampling   

*Note: All Rejection Sampling data generated with Llama 4 Maverick. This model was chosen as it is not a pure reasoning model (vs. something like DeepSeek-R1) -- we felt the outputs of this model seemed more natural language sounding given open models we had access to and could easily run on our hardware.*   

## Dataset Information 
We split `Rejection Sampling` into each of our four subtasks below. See paper for more detail but roughly:
- `Predict Move`: Given board state (no moves provided), play a move. Moves in the top 30% in move rank per Stockfish are kept.
- `Best Move / Worst Move`: Given board state and list of ~5 moves, choose the best (or worst) move. Only correct answers are kept.
- `Legal Moves`: Given board state and piece, produce a list of all legal moves that piece could make. Scored via intersection over union, only those exceeding a threshold are kept.

### General Information
| Dataset       | Num Samples | Mean Tok. Len. (Std. Dev.) | Max Tok. Len. |
|---------------|------------:|---------------------------:|--------------:|
| Predict Move  | 5,753       | 680.3 (150.4)              | 1194          |
| Best Move     | 2,526       | 739.3 (142.0)              | 1200          |
| Worst Move    | 1,612       | 794.5 (173.2)              | 1196          |
| Legal Moves   | 592         | 652.2 (213.9)              | 1197          |
| **Total**     | **10,483**  | **710.5 (162.6)**          | **1200**      |

### Piece Information (Generated Move / Candidate Move)
| Dataset       | % White | % Black | King  | Queen | Rook | Bishop | Knight | Pawn |
|---------------|--------:|--------:|------:|------:|-----:|-------:|-------:|-----:|
| Predict Move  | 52.7%   | 47.3%   | 21.0% | 21.8% | 19.3% | 3.0%  | 20.5% | 14.4% |
| Best Move     | 50.6%   | 49.4%   | 9.9%  | 20.7% | 20.9% | 16.8% | 21.1% | 10.5% |
| Worst Move    | 47.0%   | 53.0%   | 8.2%  | 22.7% | 21.4% | 22.9% | 13.5% | 11.4% |
| Legal Moves   | 57.4%   | 42.6%   | 20.1% | 7.4%  | 24.5% | 12.5% | 24.7% | 10.8% |
| **Total**     | **51.6%** | **48.4%** | **16.3%** | **20.9%** | **20.3%** | **9.9%** | **19.8%** | **12.8%** |

### Distribution by Full Move Count (i.e., Game Stage)
| Dataset       | 0-9   | 10-19 | 20-29 | 30-39 | 40+   |
|---------------|------:|------:|------:|------:|------:|
| Predict Move  | 10.0% | 30.0% | 29.9% | 15.0% | 15.0% |
| Best Move     | 10.5% | 36.9% | 29.6% | 15.1% | 7.9%  |
| Worst Move    | 11.7% | 40.0% | 26.0% | 13.1% | 9.2%  |
| Legal Moves   | 21.5% | 31.4% | 25.8% | 10.6% | 10.6% |
| **Total**     | **11.1%** | **33.3%** | **29.0%** | **14.5%** | **12.2%** |

*Note: All responses are unique, though there may be multiple different responses for the same FEN board. Token counts per Qwen2.5 tokenizer.* 

### Additional Columns: Predict Move, Best Move, and Worst Move   
| Column | Meaning |
| --- | --- |
| `generated_answer` | Move extracted from the assistant response provided in answer tags. |
| `color` | Color of the piece at the source square of `generated_answer`. |
| `piece_type` | Piece type at the source square of `generated_answer`. |

### Additional Columns: Legal Moves   
| Column | Meaning |
| --- | --- |
| `generated_answer` | List of legal moves extracted from the assistant response. |
| `color` | Piece color of the candidate move in the question. |
| `piece_type` | Piece type of the candidate move in the question. |


# Verbalized Alpha-Beta Pruning (VABP)   

## Dataset Information
### General Information
| Dataset              | Num Samples | Mean Tok. Len. (Std. Dev.) | Max Tok. Len. |
|----------------------|------------:|---------------------------:|--------------:|
| VABP                 | 10,000      | 417.3 (117.4)              | 901           |

### Piece Information (Final Answer Move)
| Dataset              | % White | % Black | King  | Queen | Rook | Bishop | Knight | Pawn |
|----------------------|--------:|--------:|------:|------:|-----:|-------:|-------:|-----:|
| VABP                 | 49.3%   | 50.7%   | 13.6% | 15.6% | 18.3% | 14.1% | 13.1% | 25.3% |

### Distribution by Full Move Count (i.e., Game Stage)
| Dataset              | 0-9   | 10-19 | 20-29 | 30-39 | 40+   |
|----------------------|------:|------:|------:|------:|------:|
| VABP                 | 16.7% | 31.6% | 25.1% | 15.5% | 11.1% |

*Note: All responses are unique, though there may be multiple different responses for the same FEN board. Token counts per Qwen2.5 tokenizer.* 

### Additional Columns   
| Column | Meaning |
| --- | --- |
| `generated_answer` | Move from the end of the assistant response provided in answer tags. |
| `color` | Color of the piece from source square of `generated_answer`. |
| `piece_type` | Piece type from source square of `generated_answer`. |


# Guided Synthetic   
*Note: `Blunders` is generated primarily using Llama 4 Maverick; `Reasonable Moves` was generated using gpt-oss-120b (low).*   

## Dataset Information   
### General Information
| Dataset           | Num Samples | Mean Tok. Len. (Std. Dev.) | Max Tok. Len. |
|-------------------|------------:|---------------------------:|--------------:|
| Blunders          | 11,000      | 54.8 (17.9)                | 394           |
| Reasonable Moves  | 49,071      | 142.4 (30.8)               | 393           |
| **Total**         | **60,071**  | **126.4 (44.5)**           | **394**       |

### Piece Information (Candidate Move)
| Dataset           | % White | % Black | King  | Queen | Rook | Bishop | Knight | Pawn |
|-------------------|--------:|--------:|------:|------:|-----:|-------:|-------:|-----:|
| Blunders          | 50.5%   | 49.5%   | 11.2% | 19.1% | 19.5% | 20.0% | 15.5% | 14.7% |
| Reasonable Moves  | 51.5%   | 48.5%   | 10.8% | 17.0% | 17.9% | 15.2% | 15.2% | 23.9% |
| **Total**         | **51.3%** | **48.7%** | **10.9%** | **17.4%** | **18.2%** | **16.1%** | **15.2%** | **22.2%** |

### Distribution by Full Move Count (i.e., Game Stage)
| Dataset           | 0-9   | 10-19 | 20-29 | 30-39 | 40+   |
|-------------------|------:|------:|------:|------:|------:|
| Blunders          | 21.6% | 29.5% | 22.5% | 14.5% | 11.9% |
| Reasonable Moves  | 10.5% | 35.6% | 25.1% | 14.9% | 13.9% |
| **Total**         | **12.6%** | **34.5%** | **24.6%** | **14.8%** | **13.6%** |

*Note: Token counts per Qwen2.5 tokenizer. For context, `Blunders` refers to moves that are clearly bad like hanging your queen; `Reasonable Moves` refers to moves that are at least reasonable -- may not be the best move but reasonable to consider. The intent was to include in training data some bad samples as we felt the model may struggle from optimistic data (i.e., learns a 'all moves are good bias' given it largely sees good moves).* 

## Additional Columns   
| Column | Meaning |
| --- | --- |
| `candidate_move` | Move proposed in the prompt, extracted from the question text. |
| `color` | Color of the piece at the source square of `candidate_move`. |
| `piece_type` | Piece type at the source square of `candidate_move`. |


# Factual Board Answering (FBA)   

## Dataset Information
### General Information
| Dataset         | Num Samples | Mean Tok. Len. (Std. Dev.) | Max Tok. Len. | Mean Questions / Sample (Std. Dev.) |
|-----------------|------------:|---------------------------:|--------------:|------------------------------------:|
| Multi-Question  | 500,000     | 8.4 (2.0)                  | 16            | 5.0 (0.8)                           |

### Distribution by Full Move Count (i.e., Game Stage)
| Dataset         | 0-9   | 10-19 | 20-29 | 30-39 | 40+   |
|-----------------|------:|------:|------:|------:|------:|
| Multi-Question  | 15.0% | 30.0% | 25.0% | 20.0% | 10.0% |

### Question Type Mix
| Dataset         | Is Check | Is Legal | Under Attack | Material Adv. Value | Mobility | Cloze Capture |
|-----------------|---------:|---------:|-------------:|--------------------:|---------:|--------------:|
| Multi-Question  | 207,498 (8.3%) | 569,352 (22.8%) | 518,484 (20.7%) | 366,533 (14.7%) | 505,775 (20.2%) | 332,146 (13.3%) |

### Is Check
| Dataset         | Yes  | No    |
|-----------------|-----:|------:|
| Multi-Question  | 1.1% | 98.9% |

### Is Legal
| Dataset         | Yes   | No    |
|-----------------|------:|------:|
| Multi-Question  | 50.1% | 49.9% |

### Under Attack
| Dataset         | Yes   | No    |
|-----------------|------:|------:|
| Multi-Question  | 33.9% | 66.1% |

### Material Advantage Value
| Dataset         | <-100 | -100-0 | 0-100 | 100+  |
|-----------------|------:|-------:|------:|------:|
| Multi-Question  | 25.2% | 13.0%  | 36.7% | 25.1% |

### Mobility
| Dataset         | 0-1   | 2-3   | 4-5   | 6+    |
|-----------------|------:|------:|------:|------:|
| Multi-Question  | 16.0% | 20.2% | 17.1% | 46.6% |

*Note: Cloze-capture target squares span all 64 board squares, ranging from `g1` (1,081; 0.3%) to `f6` (14,857; 4.5%). The square-frequency distribution roughly resembles a 2D Gaussian over the board: hottest near the center and dissipating toward the edges. Token counts per Qwen2.5 tokenizer.*

## Additional Columns

| Column | Meaning |
| --- | --- |
| `individual_qa` | Ordered list of `(question, answer)` tuples parsed from the multi-question prompt and assistant response. |
| `qa_type` | Ordered list of task labels aligned to `individual_qa`. Current labels follow the notebook names: `is_check`, `is_legal`, `under_attack`, `mat_adv_value`, `mobility`, `cloze_capture`. |


# Best Move   

## Dataset Information   
### General Information
| Dataset    | Num Samples | Mean Tok. Len. (Std. Dev.) | Max Tok. Len. |
|------------|------------:|---------------------------:|--------------:|
| Best Move  | 14,848,802  | 4.0 (0.0)                  | 5             |

### Piece Information (Generated Move)
| Dataset    | % White | % Black | King  | Queen | Rook | Bishop | Knight | Pawn |
|------------|--------:|--------:|------:|------:|-----:|-------:|-------:|-----:|
| Best Move  | 50.0%   | 50.0%   | 14.0% | 14.6% | 16.6% | 14.3% | 15.7% | 24.8% |

### Distribution by Full Move Count (i.e., Game Stage)
| Dataset    | 0-9   | 10-19 | 20-29 | 30-39 | 40+   |
|------------|------:|------:|------:|------:|------:|
| Best Move  | 14.9% | 30.1% | 25.0% | 20.0% | 10.0% |

*Note: Token counts per Qwen2.5 tokenizer.*

### Additional Columns   
| Column | Meaning |
| --- | --- |
| `generated_answer` | Move extracted from the assistant response. |
| `color` | Color of the piece at the source square of `generated_answer`. |
| `piece_type` | Piece type at the source square of `generated_answer`. |


# Best Line   
   
## Dataset Information   
### General Information
| Dataset    | Num Samples | Mean Tok. Len. (Std. Dev.) | Max Tok. Len. |
|------------|------------:|---------------------------:|--------------:|
| Best Line  | 1,604,684   | 25.7 (3.6)                 | 34            |

### Piece Information (First Move)
| Dataset    | % White | % Black | King  | Queen | Rook | Bishop | Knight | Pawn |
|------------|--------:|--------:|------:|------:|-----:|-------:|-------:|-----:|
| Best Line  | 49.8%   | 50.2%   | 10.6% | 16.5% | 17.2% | 15.2% | 16.7% | 23.8% |

### Distribution by Full Move Count (i.e., Game Stage)
| Dataset    | 0-9   | 10-19 | 20-29 | 30-39 | 40+   |
|------------|------:|------:|------:|------:|------:|
| Best Line  | 11.2% | 34.8% | 28.5% | 16.1% | 9.4% |

*Note: Token counts per Qwen2.5 tokenizer.*

### Additional Columns
| Column | Meaning |
| --- | --- |
| `first_move` | First move in the assistant response. |
| `value` | Final evaluation token normalized to a signed centipawn string such as `+3` or `-16`, or `mate`. |
| `color` | Color of the piece at the source square of `first_move`. |
| `piece_type` | Piece type at the source square of `first_move`. |
