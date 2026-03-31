# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from typing import Optional


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def extract_search_tool_calls(context: str, mcp_parser_name: Optional[str] = None) -> list[str]:
    if not mcp_parser_name:
        patterns = [
            r"<search>(.*?)</search>",
            r"<call_tool name=(.*?)>(.*?)</call_tool>",
            r"<tool name=(.*?)>(.*?)</tool>",
        ]
        extracted_queries = []
        for pattern in patterns:
            matches = re.findall(pattern, context, re.DOTALL)
            if not matches:
                continue
            if isinstance(matches[0], tuple):
                extracted_queries.extend(match[1].strip() for match in matches if match[1].strip())
            else:
                extracted_queries.extend(match.strip() for match in matches if match.strip())
        return extracted_queries

    if mcp_parser_name in {"unified"}:
        matches = re.findall(r"<tool name=(.*?)>(.*?)</tool>", context, re.DOTALL)
    elif mcp_parser_name in {"v20250824", "dr_tulu_xml"}:
        matches = re.findall(r"<call_tool name=(.*?)>(.*?)</call_tool>", context, re.DOTALL)
        if not matches:
            matches = re.findall(r"<call_tool name=(.*?)>(.*?)</call>", context, re.DOTALL)
    else:
        return []

    return [match[1].strip() for match in matches if match[1].strip()]


def compute_format_reward(
    response: str, mcp_parser_name: Optional[str] = None, use_full_response_as_answer: bool = False
) -> float:
    if use_full_response_as_answer:
        return 1.0

    answer_match = re.search(r"<answer>.*?</answer>", response, re.DOTALL)
    answer_format_reward = 1.0 if answer_match else 0.0

    citation_match = re.search(r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>", response, re.DOTALL)
    citation_format_reward = 1.0 if citation_match else 0.0

    queries = extract_search_tool_calls(response, mcp_parser_name=mcp_parser_name)
    query_format_reward = 1.0 if queries else 0.0

    return 0.5 * answer_format_reward + 0.3 * citation_format_reward + 0.2 * query_format_reward


FORMAT_REWARD_WEIGHT = 0.2


def compute_score_em(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    mcp_parser_name = extra_info.get("mcp_parser_name")
    if mcp_parser_name is None and "<call_tool name=" in solution_str:
        mcp_parser_name = "dr_tulu_xml"

    format_reward = compute_format_reward(solution_str, mcp_parser_name=mcp_parser_name)
    weighted_format_reward = FORMAT_REWARD_WEIGHT * format_reward
    answer = extract_solution(solution_str=solution_str)
    is_correct = answer is not None and subem_check(answer, ground_truth["target"])
    reward_score = compute_score_subem(
        solution_str=solution_str,
        ground_truth=ground_truth,
        format_score=weighted_format_reward,
        method=method,
        score=score,
    )
    return {
        "score": reward_score,
        "format_reward": format_reward,
        "weighted_format_reward": weighted_format_reward,
        "accuracy_reward": float(is_correct),
    }
