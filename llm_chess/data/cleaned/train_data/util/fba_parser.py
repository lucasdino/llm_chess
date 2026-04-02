from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean, median, pstdev
from typing import Any
from typing import Pattern

from llm_chess.data.raw.utils.parsing import extract_solution


QUESTION_SPLIT_MARKER = (
    "Answer the following - if multiple questions, include a space between each answer:\n"
)

_YES_NO_ANSWER_REGEX = re.compile(r"^(?:Yes|No)$", re.IGNORECASE)
_INTEGER_ANSWER_REGEX = re.compile(
    r"^-?(?:\d+|\d{1,3}(?:,\d{3})+)$",
    re.IGNORECASE,
)
_SQUARE_ANSWER_REGEX = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)

_QA_PATTERNS: tuple[tuple[str, Pattern[str], Pattern[str]], ...] = (
    (
        "is_check",
        re.compile(
            r"^Is the (black|white) king in check(?:\s*\{'Yes', 'No'\})?\?$",
            re.IGNORECASE,
        ),
        _YES_NO_ANSWER_REGEX,
    ),
    (
        "is_legal",
        re.compile(
            r"^Can you legally play\s+"
            r"([a-h][1-8][a-h][1-8][qrbn]?|"
            r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"
            r"\s*\{'Yes', 'No'\}\?$",
            re.IGNORECASE,
        ),
        _YES_NO_ANSWER_REGEX,
    ),
    (
        "under_attack",
        re.compile(
            r"^Can your "
            r"(pawn|knight|bishop|rook|queen|king) "
            r"take their "
            r"(pawn|knight|bishop|rook|queen|king)"
            r"\s*\{'Yes', 'No'\}\?$",
            re.IGNORECASE,
        ),
        _YES_NO_ANSWER_REGEX,
    ),
    (
        "mat_adv_value",
        re.compile(
            r"^What is the material advantage for "
            r"(white|black) "
            r"\(The following are values for each piece: "
            r"Pawn=100; Knight=320; Bishop=330; Rook=500; Queen=900; King=0\. "
            r"You should respond with just an integer in the format "
            r"'#,##0' or '-#,##0'\.\)\?$",
            re.IGNORECASE,
        ),
        _INTEGER_ANSWER_REGEX,
    ),
    (
        "mobility",
        re.compile(
            r"^How many legal moves "
            r"(?:could|does) your "
            r"(pawn|knight|bishop|rook|queen|king) "
            r"at ([a-h][1-8]) "
            r"(?:currently play|have) "
            r"\(answer with an integer\)\?$",
            re.IGNORECASE,
        ),
        _INTEGER_ANSWER_REGEX,
    ),
    (
        "cloze_capture",
        re.compile(
            r"^My piece on __ could take the opponent's "
            r"(pawn|knight|bishop|rook|queen|king) "
            r"on ([a-h][1-8]) "
            r"\(answer with square of only piece that makes this statement true "
            r"-- e\.g\., 'e4'\)\.$",
            re.IGNORECASE,
        ),
        _SQUARE_ANSWER_REGEX,
    ),
)

FBA_QA_TYPES = tuple(qa_type for qa_type, _, _ in _QA_PATTERNS)
_YES_NO_QA_TYPES = frozenset({"is_check", "is_legal", "under_attack"})
_NUMERIC_HISTOGRAM_BUCKETS = {
    "mat_adv_value": ("<-100", "-100-0", "0-100", "100+"),
    "mobility": ("0-1", "2-3", "4-5", "6+"),
}


def _normalize_response_text(response: str) -> str:
    response_text = response.strip()
    if not response_text:
        raise ValueError("FBA response is empty.")

    if "<answer>" not in response_text.lower():
        return response_text

    try:
        normalized = extract_solution(response_text).strip()
    except Exception as exc:
        raise ValueError("Could not extract <answer> content from FBA response.") from exc

    if not normalized:
        raise ValueError("Extracted FBA <answer> content is empty.")

    return normalized


def extract_fba_questions(question_prompt: str) -> list[str]:
    if QUESTION_SPLIT_MARKER not in question_prompt:
        raise ValueError(
            "FBA prompt is missing the multi-question split marker."
        )

    question_block = question_prompt.split(QUESTION_SPLIT_MARKER, maxsplit=1)[1]
    questions = [line.strip() for line in question_block.splitlines() if line.strip()]

    if not questions:
        raise ValueError("FBA prompt did not contain any individual questions.")

    return questions


def _match_fba_question(question: str) -> tuple[str, Pattern[str]]:
    for qa_type, question_regex, answer_regex in _QA_PATTERNS:
        if question_regex.fullmatch(question):
            return qa_type, answer_regex

    raise ValueError(f"Unrecognized FBA question template: {question!r}")


def extract_fba_individual_qa(
    *,
    question_prompt: str,
    response: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    questions = extract_fba_questions(question_prompt)
    answers = _normalize_response_text(response).split()

    if len(questions) != len(answers):
        raise ValueError(
            "FBA question/answer count mismatch: "
            f"{len(questions)} questions vs {len(answers)} answers."
        )

    individual_qa: list[tuple[str, str]] = []
    qa_types: list[str] = []

    for question, answer in zip(questions, answers):
        qa_type, answer_regex = _match_fba_question(question)
        if answer_regex.fullmatch(answer) is None:
            raise ValueError(
                f"Invalid answer {answer!r} for FBA question type {qa_type!r}: "
                f"{question!r}"
            )

        individual_qa.append((question, answer))
        qa_types.append(qa_type)

    return individual_qa, qa_types


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return [_to_python_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _to_python_value(item) for key, item in value.items()}

    return value


def normalize_stored_fba_individual_qa(value: Any) -> list[tuple[str, str]]:
    value = _to_python_value(value)
    if _is_null_like(value):
        raise ValueError("Stored FBA individual_qa value is missing.")
    if not isinstance(value, list):
        raise ValueError(
            "Stored FBA individual_qa value must be a list-like structure."
        )

    normalized_pairs: list[tuple[str, str]] = []
    for item in value:
        item = _to_python_value(item)
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(
                f"Stored FBA individual_qa item must be length-2, got {item!r}"
            )

        question, answer = item
        if _is_null_like(question) or _is_null_like(answer):
            raise ValueError("Stored FBA individual_qa item contains null values.")

        normalized_pairs.append((str(question).strip(), str(answer).strip()))

    return normalized_pairs


def normalize_stored_fba_qa_types(value: Any) -> list[str]:
    value = _to_python_value(value)
    if _is_null_like(value):
        raise ValueError("Stored FBA qa_type value is missing.")
    if not isinstance(value, list):
        raise ValueError("Stored FBA qa_type value must be a list-like structure.")

    normalized_types: list[str] = []
    for item in value:
        if _is_null_like(item):
            raise ValueError("Stored FBA qa_type contains null values.")

        normalized_types.append(str(item).strip())

    return normalized_types


def extract_fba_row_entries(
    row: dict[str, Any],
) -> list[tuple[str, str, str]]:
    stored_individual_qa = row.get("individual_qa")
    stored_qa_type = row.get("qa_type")

    if not _is_null_like(stored_individual_qa) and not _is_null_like(stored_qa_type):
        individual_qa = normalize_stored_fba_individual_qa(stored_individual_qa)
        qa_types = normalize_stored_fba_qa_types(stored_qa_type)

        if len(individual_qa) != len(qa_types):
            raise ValueError(
                "Stored FBA individual_qa and qa_type lengths do not match: "
                f"{len(individual_qa)} vs {len(qa_types)}."
            )

        entries: list[tuple[str, str, str]] = []
        for (question, answer), qa_type in zip(individual_qa, qa_types):
            matched_qa_type, answer_regex = _match_fba_question(question)
            if matched_qa_type != qa_type:
                raise ValueError(
                    "Stored FBA qa_type does not match parsed question type: "
                    f"{qa_type!r} vs {matched_qa_type!r} for {question!r}"
                )
            if answer_regex.fullmatch(answer) is None:
                raise ValueError(
                    f"Stored FBA answer {answer!r} is invalid for {qa_type!r}"
                )

            entries.append((question, answer, qa_type))

        return entries

    question_prompt = row.get("question")
    response = row.get("response")
    if not isinstance(question_prompt, str) or not isinstance(response, str):
        raise ValueError(
            "FBA row is missing question/response and cannot be reconstructed."
        )

    individual_qa, qa_types = extract_fba_individual_qa(
        question_prompt=question_prompt,
        response=response,
    )
    return [
        (question, answer, qa_type)
        for (question, answer), qa_type in zip(individual_qa, qa_types)
    ]


def parse_fba_numeric_answer(answer: str) -> int:
    normalized = answer.replace(",", "").strip()
    if _INTEGER_ANSWER_REGEX.fullmatch(normalized) is None:
        raise ValueError(f"Expected integer-like FBA answer, got {answer!r}")

    return int(normalized)


def bucket_fba_answer(qa_type: str, answer: str) -> str:
    answer = answer.strip()

    if qa_type in _YES_NO_QA_TYPES:
        if _YES_NO_ANSWER_REGEX.fullmatch(answer) is None:
            raise ValueError(f"Expected Yes/No answer for {qa_type!r}, got {answer!r}")

        return answer.title()

    if qa_type == "mat_adv_value":
        value = parse_fba_numeric_answer(answer)
        if value < -100:
            return "<-100"
        if value < 0:
            return "-100-0"
        if value <= 100:
            return "0-100"
        return "100+"

    if qa_type == "mobility":
        value = parse_fba_numeric_answer(answer)
        if value <= 1:
            return "0-1"
        if value <= 3:
            return "2-3"
        if value <= 5:
            return "4-5"
        return "6+"

    if qa_type == "cloze_capture":
        normalized = answer.lower()
        if _SQUARE_ANSWER_REGEX.fullmatch(normalized) is None:
            raise ValueError(
                f"Expected square answer for cloze_capture, got {answer!r}"
            )

        return normalized

    raise ValueError(f"Unsupported FBA qa_type for bucketing: {qa_type!r}")


def _summary_from_numeric_values(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "std_dev": None,
        }

    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
        "std_dev": pstdev(values),
    }


def summarize_fba_entries(
    entries: list[tuple[str, str, str]],
) -> dict[str, Any]:
    num_questions = len(entries)
    qa_type_counts = Counter()
    categorical_answer_counts: dict[str, Counter[str]] = {
        qa_type: Counter() for qa_type in FBA_QA_TYPES
    }
    numeric_answers: dict[str, list[int]] = {
        qa_type: [] for qa_type in _NUMERIC_HISTOGRAM_BUCKETS
    }
    cloze_file_counts = Counter()
    cloze_rank_counts = Counter()

    for _, answer, qa_type in entries:
        qa_type_counts.update([qa_type])
        bucketed_answer = bucket_fba_answer(qa_type, answer)
        categorical_answer_counts[qa_type].update([bucketed_answer])

        if qa_type in numeric_answers:
            numeric_answers[qa_type].append(parse_fba_numeric_answer(answer))

        if qa_type == "cloze_capture":
            cloze_file_counts.update([bucketed_answer[0]])
            cloze_rank_counts.update([bucketed_answer[1]])

    qa_type_distribution = {
        qa_type: {
            "count": qa_type_counts.get(qa_type, 0),
            "percentage": (
                100.0 * qa_type_counts.get(qa_type, 0) / num_questions
                if num_questions
                else 0.0
            ),
        }
        for qa_type in FBA_QA_TYPES
    }

    answer_distributions: dict[str, Any] = {}
    for qa_type in FBA_QA_TYPES:
        counts = categorical_answer_counts[qa_type]
        question_count = qa_type_counts.get(qa_type, 0)

        distribution: dict[str, Any] = {
            "num_questions": question_count,
            "counts": dict(counts.most_common()),
            "percentages": {
                key: 100.0 * value / question_count
                for key, value in counts.most_common()
            } if question_count else {},
        }

        if qa_type in _NUMERIC_HISTOGRAM_BUCKETS:
            distribution["bucket_order"] = list(_NUMERIC_HISTOGRAM_BUCKETS[qa_type])
            distribution["numeric_summary"] = _summary_from_numeric_values(
                numeric_answers[qa_type]
            )
        elif qa_type == "cloze_capture":
            distribution["file_counts"] = dict(cloze_file_counts.most_common())
            distribution["rank_counts"] = dict(cloze_rank_counts.most_common())

        answer_distributions[qa_type] = distribution

    return {
        "num_questions": num_questions,
        "qa_type_distribution": qa_type_distribution,
        "answer_distributions": answer_distributions,
    }
