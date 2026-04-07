# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.reward_score import search_r1_like_qa_em


def test_compute_format_reward_for_dr_tulu_xml():
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<tool_output><snippet id=s1>Title: A</snippet></tool_output>'
        '<answer><cite id="s1">AI is in the news.</cite></answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(response, mcp_parser_name="dr_tulu_xml")

    assert reward == 1.0


def test_compute_score_falls_back_to_format_reward():
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<answer><cite id="s1">wrong answer</cite></answer>'
    )
    ground_truth = {"target": ["correct answer"]}

    result = search_r1_like_qa_em.compute_score(response, ground_truth)

    assert result["accuracy_reward"] == 0.0
    assert result["format_reward"] == 1.0
    assert result["weighted_format_reward"] == 0.2
    assert result["score"] == 1.0


def test_compute_format_reward_strict_requires_single_final_answer_block():
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<tool_output><snippet id="s1">Title: A</snippet></tool_output>'
        '<answer><cite id="s1">AI is in the news.</cite></answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert reward == 0.0


def test_compute_format_reward_strict_penalizes_non_final_answer_block():
    response = (
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<answer><cite id="s1">AI is in the news.</cite></answer>'
        '<think>extra trailing block</think>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert reward == -1.0


def test_compute_score_strict_applies_negative_format_penalty():
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<answer><cite id="s1">wrong answer</cite></answer>'
        '<think>extra trailing block</think>'
    )
    ground_truth = {"target": ["correct answer"]}

    result = search_r1_like_qa_em.compute_score(response, ground_truth, format_penalty="strict")

    assert result["accuracy_reward"] == 0.0
    assert result["format_reward"] == -1.0
    assert "weighted_format_reward" not in result
    assert result["score"] == -1.0
