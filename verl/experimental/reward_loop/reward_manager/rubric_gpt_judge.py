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
    """LLM-as-judge reward manager that scores each rollout independently."""

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

        # Judge returns raw score in [score_range_min, score_range_max], default [0, 10].
        # reward_score used by trainer is normalized into [0, 1].
        self.score_normalization = str(judge_cfg.get("score_normalization", "fixed_range"))
        self.score_range_min = float(judge_cfg.get("score_range_min", 0.0))
        self.score_range_max = float(judge_cfg.get("score_range_max", 10.0))
        self.equal_score_reward = float(judge_cfg.get("equal_score_reward", 0.5))
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

        # Kept in the same style as user-provided Azure OpenAI snippet.
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

        if self.strict_n_rollouts:
            logger.warning(
                "strict_n_rollouts=True is ignored in single-rollout judge mode. "
                "Each rollout is judged independently."
            )

    async def run_single(self, data: DataProto) -> dict:
        """Score a single rollout with one GPT call."""
        assert len(data) == 1, "Only support single data item"
        output = self._score_item(
            data_item=data[0],
            global_step=int(data.meta_info.get("global_steps", -1)),
            group_id=0,
            rollout_index=0,
        )
        return output

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Score each rollout independently (one rollout per prompt to GPT)."""
        outputs: list[dict] = []
        global_step = int(data.meta_info.get("global_steps", -1))
        for idx in range(len(data)):
            output = self._score_item(
                data_item=data[idx],
                global_step=global_step,
                group_id=idx,
                rollout_index=0,
            )
            outputs.append(output)
        return outputs

    def _score_item(self, data_item: DataProto, global_step: int, group_id: int, rollout_index: int) -> dict:
        query, rubrics = self._extract_query_and_rubrics(data_item)
        rollout_text = self._extract_response_text(data_item)
        judge_prompt = self._build_judge_prompt(query=query, rubrics=rubrics, rollout_text=rollout_text)
        judge_raw_output = ""

        try:
            judge_result, judge_raw_output = self._judge_rollout(judge_prompt=judge_prompt)
            raw_score = self._extract_single_score(judge_result)
            normalized_score = self._normalize_single_score(raw_score)
            reason = str(judge_result.get("reason", "")).strip()
        except Exception as exc:
            if not self.fallback_to_first_on_error:
                raise
            reason = f"fallback due to judge error: {exc}"
            raw_score = self.score_range_min
            normalized_score = self.lose_score
            logger.warning("Judge failed for group=%s rollout=%s, fallback to lose_score. error=%s", group_id, rollout_index, exc)

        uid = str(data_item.non_tensor_batch.get("uid", ""))
        self._append_rollout_record(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "uid": uid,
                "group_id": group_id,
                "rollout_index": rollout_index,
                "expected_n_rollouts": self.n_rollouts,
                "query": query,
                "rubrics": rubrics,
                "response": rollout_text,
                "judge_prompt": judge_prompt,
                "judge_raw_output": judge_raw_output,
                "raw_score": float(raw_score),
                "normalized_score": float(normalized_score),
                "reason": reason,
                "judge_model": self.model,
            }
        )

        return {
            "reward_score": float(normalized_score),
            "reward_extra_info": {
                "acc": float(normalized_score),
                "gpt_judge_score": float(normalized_score),
                "gpt_judge_raw_score": float(raw_score),
                "gpt_judge_normalized_score": float(normalized_score),
                "gpt_judge_reason": reason,
                "gpt_judge_group_id": group_id,
                "gpt_judge_rollout_index": rollout_index,
            },
        }

    def _append_rollout_record(self, record: dict[str, Any]) -> None:
        if self.rollout_save_path is None:
            return
        with self._io_lock:
            with self.rollout_save_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_query_and_rubrics(self, data_item: DataProto) -> tuple[str, list[dict[str, Any]]]:
        ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {}) or {}
        query = str(ground_truth.get("query", "")).strip() or self._extract_prompt_text(data_item)
        rubrics = ground_truth.get("rubrics", [])
        if not isinstance(rubrics, list):
            rubrics = []
        return query, rubrics

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

    def _judge_rollout(self, judge_prompt: str) -> tuple[dict[str, Any], str]:
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict deep-research scoring expert. "
                                "Return only valid JSON without markdown."
                            ),
                        },
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = resp.choices[0].message.content or ""
                return self._parse_judge_output(content), content
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)

        raise RuntimeError(last_error)

    def _parse_judge_output(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_markdown_json_fence(text)

        # Primary path: strict JSON object.
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        # Tolerate plain numeric outputs.
        if isinstance(parsed, (int, float)):
            return {"score": float(parsed)}

        try:
            return {"score": float(cleaned)}
        except Exception as exc:
            raise ValueError(f"Judge output is not valid JSON score: {cleaned}") from exc

    def _extract_single_score(self, judge_result: dict[str, Any]) -> float:
        if "score" in judge_result:
            return float(judge_result["score"])
        if "final_score" in judge_result:
            return float(judge_result["final_score"])
        scores = judge_result.get("scores")
        if isinstance(scores, list) and len(scores) > 0:
            return float(scores[0])
        raise ValueError(f"Judge JSON missing score field: {judge_result}")

    def _normalize_single_score(self, raw_score: float) -> float:
        clipped = min(self.score_range_max, max(self.score_range_min, float(raw_score)))
        denom = self.score_range_max - self.score_range_min
        if denom <= 1e-12:
            return float(self.equal_score_reward)

        # Single-rollout scoring has no meaningful group min-max; use fixed-range mapping.
        if self.score_normalization not in {"fixed_range", "group_minmax"}:
            logger.warning("Unknown score_normalization=%s, fallback to fixed_range.", self.score_normalization)
        return min(1.0, max(0.0, (clipped - self.score_range_min) / denom))

    @staticmethod
    def _strip_markdown_json_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return cleaned.strip()

    def _build_judge_prompt(self, query: str, rubrics: list[dict[str, Any]], rollout_text: str) -> str:
        rubric_lines = []
        for i, item in enumerate(rubrics):
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", item.get("rubric", ""))).strip()
            weight = item.get("weight", 1)
            rubric_lines.append(
                f"{i + 1}. title: {title}\n"
                f"   description: {description}\n"
                f"   weight: {weight}"
            )
        rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."

        return (
            "You are an expert scorer for deep-research quality.\n"
            "For the query and response below, score ONLY this single response using the rubric.\n\n"
            "Query:\n"
            f"{query}\n\n"
            "Response:\n"
            f"{rollout_text}\n\n"
            "Rubric (title, description, weight):\n"
            f"{rubric_block}\n\n"
            "Scoring rule:\n"
            "- Return one final score in [0, 10], where 0 is worst and 10 is best.\n"
            "- Weight indicates relative importance among rubric items, not a normalized coefficient.\n"
            "- Do not output explanations.\n"
            "- Output must be JSON only, with this schema:\n"
            '{\n  "score": 7.5\n}\n'
        )
