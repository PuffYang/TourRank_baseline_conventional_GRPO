# Copyright 2026 Individual Contributor
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

import json
import time
from collections import OrderedDict
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

DEFAULT_AZURE_OPENAI_API_KEY = "b1b6cfd6240c446dbbe8ca087ca7fc02"
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-06-01"
DEFAULT_AZURE_OPENAI_ENDPOINT = "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-06-01"


def _strip_markdown_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


@register("rubric_judge")
class RubricJudgeRewardManager(RewardManagerBase):
    """Rubric-aware group judge reward manager for GRPO.

    This manager groups rollouts by ``uid`` and asks GPT-4o to judge all
    responses in a group jointly using query-specific rubrics.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        judge_cfg = config.reward.get("rubric_judge", {})

        self.model = judge_cfg.get("model", "gpt-4o")
        self.temperature = float(judge_cfg.get("temperature", 0.0))
        self.max_tokens = int(judge_cfg.get("max_tokens", 1200))
        self.timeout = int(judge_cfg.get("timeout", 200))
        self.max_retries = int(judge_cfg.get("max_retries", 3))
        self.retry_sleep = float(judge_cfg.get("retry_sleep", 1.5))
        self.n_rollouts = int(judge_cfg.get("n_rollouts", config.actor_rollout_ref.rollout.n))

        api_key = judge_cfg.get("api_key", DEFAULT_AZURE_OPENAI_API_KEY)
        api_version = judge_cfg.get("api_version", DEFAULT_AZURE_OPENAI_API_VERSION)
        azure_endpoint = judge_cfg.get("azure_endpoint", DEFAULT_AZURE_OPENAI_ENDPOINT)

        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("openai package is required for rubric_judge reward manager. Please `pip install openai`.") from e

        # Keep the same calling style as the provided AzureOpenAI snippet.
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

    async def run_single(self, data: DataProto) -> dict:
        outputs = await self.run_batch(data)
        assert len(outputs) == 1, f"run_single expects exactly one output, got {len(outputs)}"
        return outputs[0]

    async def run_batch(self, data: DataProto) -> list[dict]:
        if len(data) == 0:
            return []

        responses = self._decode_responses(data)
        uid_groups: OrderedDict[str, list[int]] = OrderedDict()
        if "uid" in data.non_tensor_batch:
            for idx, uid in enumerate(data.non_tensor_batch["uid"]):
                uid_groups.setdefault(str(uid), []).append(idx)
        else:
            for idx in range(len(data)):
                uid_groups[str(idx)] = [idx]

        outputs: list[dict | None] = [None] * len(data)

        for group_id, (_, indices) in enumerate(uid_groups.items()):
            data_item = data[indices[0]]
            query, rubrics = self._extract_query_and_rubrics(data_item.non_tensor_batch)

            rollout_payload = []
            for local_rollout_id, idx in enumerate(indices):
                rollout_payload.append({"rollout_id": local_rollout_id, "response": responses[idx]})

            judge_result = await self.loop.run_in_executor(
                None,
                lambda: self._judge_group(
                    group_id=group_id,
                    query=query,
                    rubrics=rubrics,
                    rollouts=rollout_payload,
                ),
            )
            score_map, winners, reason = self._parse_judge_result(judge_result, len(indices))

            if len(indices) != self.n_rollouts:
                reason = (
                    f"{reason} [group_size={len(indices)} differs from configured n_rollouts={self.n_rollouts}]".strip()
                )

            winners_json = json.dumps(winners, ensure_ascii=False)
            for local_rollout_id, idx in enumerate(indices):
                score = score_map.get(local_rollout_id)
                if score is None:
                    score = 1.0 if local_rollout_id in winners else 0.0
                outputs[idx] = {
                    "reward_score": float(score),
                    "reward_extra_info": {
                        "acc": float(score),
                        "judge_group_id": int(group_id),
                        "judge_reason": str(reason),
                        "judge_winners": winners_json,
                    },
                }

        assert all(item is not None for item in outputs), "Some rubric_judge rewards are missing."
        return outputs  # type: ignore[return-value]

    def _decode_responses(self, data: DataProto) -> list[str]:
        prompt_len = data.batch["prompts"].shape[-1]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

        responses = []
        for i in range(len(data)):
            valid_len = int(valid_response_lengths[i].item())
            response_ids = data.batch["responses"][i][:valid_len]
            responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        return responses

    def _extract_query_and_rubrics(self, non_tensor_batch: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        query = None
        rubrics = None

        reward_model = non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
            if isinstance(ground_truth, dict):
                query = ground_truth.get("query")
                rubrics = ground_truth.get("rubrics")

        extra_info = non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict):
            if query is None:
                query = extra_info.get("query")
            if rubrics is None:
                rubrics = extra_info.get("rubrics")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("rubric_judge requires query in reward_model.ground_truth.query or extra_info.query")
        if not isinstance(rubrics, list) or len(rubrics) == 0:
            raise ValueError(
                "rubric_judge requires rubrics list in reward_model.ground_truth.rubrics or extra_info.rubrics"
            )

        normalized_rubrics = []
        for rubric in rubrics:
            if not isinstance(rubric, dict):
                continue
            title = str(rubric.get("title", "")).strip()
            description = str(rubric.get("description", "")).strip()
            weight = rubric.get("weight", 1)
            try:
                weight = int(weight)
            except Exception:
                weight = 1
            normalized_rubrics.append(
                {
                    "title": title,
                    "description": description,
                    "weight": weight,
                }
            )

        if len(normalized_rubrics) == 0:
            raise ValueError("rubric_judge failed to parse non-empty rubric items")

        return query.strip(), normalized_rubrics

    def _judge_group(
        self,
        group_id: int,
        query: str,
        rubrics: list[dict[str, Any]],
        rollouts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "group_id": group_id,
            "query": query,
            "rubrics": rubrics,
            "rollouts": rollouts,
        }
        user_prompt = (
            "You are an expert evaluator for deep research and long-form answers.\n\n"
            "Judge all candidate rollouts for the same query using the provided rubric.\n"
            "You MUST explicitly use each rubric item's title, description, and weight.\n"
            "Weight indicates relative importance (not normalized coefficient), so higher-weight items should matter more.\n"
            "For each rollout, output one final total score.\n\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "group_id": <int>,\n'
            '  "scores": [{"rollout_id": <int>, "total_score": <number>}],\n'
            '  "winners": [<int>, ...],\n'
            '  "reason": "<short reason>"\n'
            "}\n\n"
            "Input:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict JSON generator. Return only valid JSON, no markdown.",
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = resp.choices[0].message.content or ""
                result = json.loads(_strip_markdown_json_fence(content))
                if not isinstance(result, dict):
                    raise ValueError("Judge output is not a JSON object.")
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)
        raise RuntimeError(last_error)

    def _parse_judge_result(
        self, judge_result: dict[str, Any], group_size: int
    ) -> tuple[dict[int, float], list[int], str]:
        scores_by_rollout: dict[int, float] = {}
        winners: list[int] = []
        reason = str(judge_result.get("reason", ""))

        raw_scores = judge_result.get("scores", [])
        if isinstance(raw_scores, list):
            for i, item in enumerate(raw_scores):
                if isinstance(item, dict):
                    rollout_id = item.get("rollout_id", i)
                    score = item.get("total_score", item.get("score", 0.0))
                else:
                    rollout_id = i
                    score = item

                try:
                    rollout_id = int(rollout_id)
                    score = float(score)
                except Exception:
                    continue

                if 0 <= rollout_id < group_size:
                    scores_by_rollout[rollout_id] = score

        raw_winners = judge_result.get("winners", [])
        if isinstance(raw_winners, list):
            parsed_winners = []
            for w in raw_winners:
                try:
                    w_int = int(w)
                except Exception:
                    continue
                if 0 <= w_int < group_size:
                    parsed_winners.append(w_int)
            winners = sorted(set(parsed_winners))

        if not winners and scores_by_rollout:
            max_score = max(scores_by_rollout.values())
            winners = sorted([rid for rid, score in scores_by_rollout.items() if score == max_score])

        return scores_by_rollout, winners, reason
