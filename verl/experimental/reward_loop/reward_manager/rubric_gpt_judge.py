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
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from openai import AzureOpenAI
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("rubric_gpt_judge")
class RubricGPTJudgeRewardManager(RewardManagerBase):
    """Group-wise GRPO reward manager using rubric-based GPT judge."""

    def __init__(self, config: DictConfig, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        judge_cfg = config.reward.get("gpt_judge", {})

        self.model = judge_cfg.get("model", "gpt-4o")
        self.temperature = float(judge_cfg.get("temperature", 0.0))
        self.max_tokens = int(judge_cfg.get("max_tokens", 1200))
        self.timeout = int(judge_cfg.get("timeout", 200))
        self.max_retries = int(judge_cfg.get("max_retries", 3))
        self.retry_sleep = float(judge_cfg.get("retry_sleep", 1.5))

        self.n_rollouts = int(judge_cfg.get("n_rollouts", config.actor_rollout_ref.rollout.n))
        self.strict_n_rollouts = bool(judge_cfg.get("strict_n_rollouts", False))
        self.max_rollout_chars = int(judge_cfg.get("max_rollout_chars", 12000))
        self.fallback_to_first_on_error = bool(judge_cfg.get("fallback_to_first_on_error", True))

        # Judge returns per-rollout raw scores (default expected range: [0, 10]).
        # reward_score used by trainer is normalized into [0, 1].
        self.score_normalization = str(judge_cfg.get("score_normalization", "group_minmax"))
        self.score_range_min = float(judge_cfg.get("score_range_min", 0.0))
        self.score_range_max = float(judge_cfg.get("score_range_max", 10.0))
        self.equal_score_reward = float(judge_cfg.get("equal_score_reward", 0.5))

        self.win_score = float(judge_cfg.get("win_score", 1.0))
        self.lose_score = float(judge_cfg.get("lose_score", 0.0))

        api_key = judge_cfg.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = judge_cfg.get("api_version", "2024-06-01")
        azure_endpoint = judge_cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise ValueError("Missing Azure OpenAI API key. Set reward.gpt_judge.api_key or AZURE_OPENAI_API_KEY.")
        if not azure_endpoint:
            raise ValueError(
                "Missing Azure OpenAI endpoint. Set reward.gpt_judge.azure_endpoint or AZURE_OPENAI_ENDPOINT."
            )
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

        rollout_save_dir = judge_cfg.get("rollout_save_dir")
        self.rollout_save_path = None
        self._io_lock = Lock()
        if rollout_save_dir:
            rollout_dir = Path(rollout_save_dir)
            rollout_dir.mkdir(parents=True, exist_ok=True)
            self.rollout_save_path = rollout_dir / f"gpt_judge_rollouts_worker_{os.getpid()}.jsonl"

    async def run_single(self, data: DataProto) -> dict:
        """Fallback single-sample scoring."""
        assert len(data) == 1, "Only support single data item"
        return {
            "reward_score": 1.0,
            "reward_extra_info": {
                "acc": 1.0,
                "gpt_judge_score": 1.0,
                "gpt_judge_raw_score": self.score_range_max,
                "gpt_judge_normalized_score": 1.0,
                "gpt_judge_winners": json.dumps([0], ensure_ascii=False),
                "gpt_judge_reason": "single rollout fallback (score mode)",
                "gpt_judge_group_id": 0,
                "gpt_judge_rollout_index": 0,
            },
        }

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Compute group-wise rewards by judging n rollouts from the same uid together."""
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        uids = data.non_tensor_batch.get("uid", [])
        for idx, uid in enumerate(uids):
            grouped_indices[str(uid)].append(idx)

        outputs: list[dict[str, Any] | None] = [None for _ in range(len(data))]
        for group_id, (uid, indices) in enumerate(grouped_indices.items()):
            if self.strict_n_rollouts and len(indices) != self.n_rollouts:
                raise ValueError(
                    f"Group uid={uid} has {len(indices)} rollouts, expected n_rollouts={self.n_rollouts}."
                )

            first_item = data[indices[0]]
            ground_truth = first_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {}) or {}
            query = str(ground_truth.get("query", "")).strip() or self._extract_prompt_text(first_item)
            rubrics = ground_truth.get("rubrics", [])
            rollout_texts = [self._extract_response_text(data[idx]) for idx in indices]

            judge_result = self._judge_group(
                group_id=group_id,
                query=query,
                rubrics=rubrics,
                rollout_texts=rollout_texts,
            )
            raw_scores = self._extract_group_scores(judge_result=judge_result, num_rollouts=len(rollout_texts))
            normalized_scores = self._normalize_group_scores(raw_scores)

            winners = self._normalize_winners(judge_result.get("winners", []), len(rollout_texts))
            if not winners:
                max_score = max(normalized_scores)
                winners = [i for i, s in enumerate(normalized_scores) if abs(s - max_score) < 1e-8]
            reason = str(judge_result.get("reason", "")).strip()

            group_rollout_records = []
            for local_idx, data_idx in enumerate(indices):
                reward = float(normalized_scores[local_idx])
                outputs[data_idx] = {
                    "reward_score": reward,
                    "reward_extra_info": {
                        "acc": reward,
                        "gpt_judge_score": reward,
                        "gpt_judge_raw_score": float(raw_scores[local_idx]),
                        "gpt_judge_normalized_score": reward,
                        "gpt_judge_winners": json.dumps(winners, ensure_ascii=False),
                        "gpt_judge_reason": reason,
                        "gpt_judge_group_id": group_id,
                        "gpt_judge_rollout_index": local_idx,
                    },
                }
                group_rollout_records.append(
                    {
                        "rollout_index": local_idx,
                        "response": rollout_texts[local_idx],
                        "raw_score": float(raw_scores[local_idx]),
                        "reward": float(reward),
                    }
                )

            self._append_rollout_record(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "global_step": int(data.meta_info.get("global_steps", -1)),
                    "uid": uid,
                    "group_id": group_id,
                    "n_rollouts": len(indices),
                    "query": query,
                    "rubrics": rubrics,
                    "winners": winners,
                    "raw_scores": [float(x) for x in raw_scores],
                    "normalized_scores": [float(x) for x in normalized_scores],
                    "reason": reason,
                    "rollouts": group_rollout_records,
                    "judge_model": self.model,
                }
            )

        # Fallback for unexpected empty positions
        for i, item in enumerate(outputs):
            if item is None:
                outputs[i] = {
                    "reward_score": self.lose_score,
                    "reward_extra_info": {
                        "acc": self.lose_score,
                        "gpt_judge_score": self.lose_score,
                        "gpt_judge_raw_score": self.score_range_min,
                        "gpt_judge_normalized_score": self.lose_score,
                        "gpt_judge_winners": "[]",
                        "gpt_judge_reason": "fallback: missing group output",
                        "gpt_judge_group_id": -1,
                        "gpt_judge_rollout_index": -1,
                    },
                }

        return outputs

    def _append_rollout_record(self, record: dict[str, Any]) -> None:
        if self.rollout_save_path is None:
            return
        with self._io_lock:
            with self.rollout_save_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_prompt_text(self, data_item: DataProto) -> str:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
        valid_prompt_ids = prompt_ids[-valid_prompt_length:].tolist()
        return self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

    def _extract_response_text(self, data_item: DataProto) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length].tolist()
        response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        if len(response) > self.max_rollout_chars:
            response = response[: self.max_rollout_chars]
        return response

    def _judge_group(self, group_id: int, query: str, rubrics: list[dict[str, Any]], rollout_texts: list[str]) -> dict[str, Any]:
        prompt = self._build_judge_prompt(group_id=group_id, query=query, rubrics=rubrics, rollout_texts=rollout_texts)

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict JSON judge for ranking multiple candidate answers. "
                                "Return only valid JSON without markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = resp.choices[0].message.content or ""
                parsed = json.loads(self._strip_markdown_json_fence(content))
                if not isinstance(parsed, dict):
                    raise ValueError("Judge output must be a JSON object.")
                return parsed
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)

        if self.fallback_to_first_on_error:
            logger.warning("Judge failed for group %s after retries, fallback to first winner. error=%s", group_id, last_error)
            scores = [self.score_range_min for _ in rollout_texts]
            if scores:
                scores[0] = self.score_range_max
            return {
                "group_id": group_id,
                "scores": scores,
                "winners": [0],
                "reason": f"fallback due to judge error: {last_error}",
            }
        raise RuntimeError(last_error)

    @staticmethod
    def _normalize_winners(winners: Any, num_rollouts: int) -> list[int]:
        if not isinstance(winners, list):
            return []

        normalized: list[int] = []
        for item in winners:
            try:
                idx = int(item)
            except Exception:
                continue
            normalized.append(idx)

        if not normalized:
            return []

        # Support both 0-based and 1-based indices.
        if all(1 <= idx <= num_rollouts for idx in normalized) and 0 not in normalized:
            normalized = [idx - 1 for idx in normalized]

        deduped = []
        seen = set()
        for idx in normalized:
            if 0 <= idx < num_rollouts and idx not in seen:
                seen.add(idx)
                deduped.append(idx)

        return deduped

    def _extract_group_scores(self, judge_result: dict[str, Any], num_rollouts: int) -> list[float]:
        raw_scores = judge_result.get("scores")
        if isinstance(raw_scores, list):
            # Case 1: dense numeric list with same order as rollout indices.
            if len(raw_scores) == num_rollouts and all(not isinstance(x, dict) for x in raw_scores):
                dense_scores = []
                for item in raw_scores:
                    try:
                        dense_scores.append(float(item))
                    except Exception:
                        dense_scores.append(self.score_range_min)
                return dense_scores

            # Case 2: sparse list of {"index": i, "score": s}.
            indexed_scores = [self.score_range_min for _ in range(num_rollouts)]
            any_indexed = False
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                    score = float(item.get("score"))
                except Exception:
                    continue
                if 0 <= idx < num_rollouts:
                    indexed_scores[idx] = score
                    any_indexed = True
            if any_indexed:
                return indexed_scores

        # Backward-compatible fallback: derive raw scores from winners.
        winners = self._normalize_winners(judge_result.get("winners", []), num_rollouts)
        scores = [self.score_range_min for _ in range(num_rollouts)]
        for idx in winners:
            scores[idx] = self.score_range_max
        if not winners and scores:
            scores[0] = self.score_range_max
        return scores

    def _normalize_group_scores(self, raw_scores: list[float]) -> list[float]:
        if not raw_scores:
            return []

        clipped = [min(self.score_range_max, max(self.score_range_min, float(s))) for s in raw_scores]

        if self.score_normalization == "fixed_range":
            denom = self.score_range_max - self.score_range_min
            if denom <= 1e-12:
                return [self.equal_score_reward for _ in clipped]
            return [min(1.0, max(0.0, (s - self.score_range_min) / denom)) for s in clipped]

        # Default: group-wise min-max normalization.
        s_min = min(clipped)
        s_max = max(clipped)
        if s_max - s_min <= 1e-12:
            return [self.equal_score_reward for _ in clipped]
        return [(s - s_min) / (s_max - s_min) for s in clipped]

    @staticmethod
    def _strip_markdown_json_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return cleaned.strip()

    def _build_judge_prompt(self, group_id: int, query: str, rubrics: list[dict[str, Any]], rollout_texts: list[str]) -> str:
        rubric_lines = []
        for i, item in enumerate(rubrics):
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            weight = item.get("weight", 1)
            rubric_lines.append(
                f"{i + 1}. title: {title}\n"
                f"   description: {description}\n"
                f"   weight: {weight}"
            )

        rollout_lines = []
        for i, text in enumerate(rollout_texts):
            rollout_lines.append(f"[{i}]\n{text}")

        rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."
        rollout_block = "\n\n".join(rollout_lines)

        return (
            "You are evaluating multiple rollout answers for the same user query.\n\n"
            f"Group ID: {group_id}\n\n"
            "User query:\n"
            f"{query}\n\n"
            "Rubric items (weight indicates relative importance, not normalized coefficient):\n"
            f"{rubric_block}\n\n"
            "Candidate rollouts:\n"
            f"{rollout_block}\n\n"
            "Task:\n"
            "1) Compare all candidates using all rubric items.\n"
            "2) Pay more attention to higher-weight rubric items.\n"
            "3) Return a score for EACH candidate rollout.\n"
            "4) Score range must be 0 to 10 (higher is better).\n"
            "5) Also return winners as candidate indices that tie for best score.\n\n"
            "Return only JSON with this schema:\n"
            "{\n"
            f'  "group_id": {group_id},\n'
            '  "scores": [7.8, 6.2, 9.1, 5.4],\n'
            '  "winners": [0],\n'
            '  "reason": "short explanation"\n'
            "}\n"
        )
