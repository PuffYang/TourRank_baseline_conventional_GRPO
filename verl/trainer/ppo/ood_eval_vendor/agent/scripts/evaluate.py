#!/usr/bin/env python3
"""
Lightweight evaluation entrypoint vendored into verl for OOD validation.

This script intentionally supports only the benchmark tasks used by verl's
OOD validation pipeline: `healthbench` and `researchqa`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(AGENT_ROOT))
sys.path.append(str(AGENT_ROOT / "evaluation"))

from samplers import common

from evaluation.health_bench_eval.healthbench_eval import HealthBenchEval
from evaluation.research_qa_eval.compute_coverage import compute_coverage, compute_coverage_score
from evaluation.research_qa_eval.researchqa_loader import download_researchqa_dataset, load_researchqa_data
from evaluation.shared_azure_gpt4o import resolve_azure_gpt4o_model
from evaluation.samplers.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def load_jsonl(file_path: str) -> list[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_to_evaluate_format(original_examples: list[dict]) -> list[dict]:
    converted_examples = []
    for ele in original_examples:
        converted_examples.append(
            {
                "id": ele["example_id"],
                "row": ele["original_data"],
                "response_text": ele["final_response"],
                "actual_queried_prompt_messages": [{"content": ele["problem"], "role": "user"}],
            }
        )
    return converted_examples


def _aggregate_researchqa_metrics(
    dataset_path: str,
    rubric_results: dict[str, list[dict]],
) -> dict[str, object]:
    items = load_researchqa_data(dataset_path)
    item_by_id = {item.id: item for item in items}

    scored_coverages: dict[str, float] = {}
    by_general_domain: dict[str, list[float]] = defaultdict(list)
    by_subdomain: dict[str, list[float]] = defaultdict(list)
    by_field: dict[str, list[float]] = defaultdict(list)

    for item_id, judges in rubric_results.items():
        if item_id not in item_by_id or not judges:
            continue

        coverage = float(compute_coverage_score(judges))
        scored_coverages[item_id] = coverage

        item = item_by_id[item_id]
        by_general_domain[item.general_domain].append(coverage)
        by_subdomain[item.subdomain].append(coverage)
        by_field[item.field].append(coverage)

    def _summarize(groups: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for name, values in sorted(groups.items()):
            if not values:
                continue
            summary[name] = {
                "coverage": float(sum(values) / len(values)),
                "count": float(len(values)),
            }
        return summary

    metrics: dict[str, object] = {
        "num_scored_examples": float(len(scored_coverages)),
        "by_general_domain": _summarize(by_general_domain),
        "by_subdomain": _summarize(by_subdomain),
        "by_field": _summarize(by_field),
    }

    return metrics


def save_evaluation_results(results_path: str, task_name: str, final_result, generation_data: list[dict]) -> None:
    results_data = {
        "task": task_name,
        "run_mode": "evaluation",
        "score": final_result.score,
        "metrics": final_result.metrics,
        "metadata": final_result.metadata,
        "num_examples": len(generation_data),
    }
    if final_result.per_example_results:
        results_data["per_example_results"] = final_result.per_example_results

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str)


def evaluate_healthbench(
    file_path: str,
    save_path: str | None,
    grader_model: str,
    n_threads: int,
) -> None:
    original_examples = load_jsonl(file_path)
    converted_examples = convert_to_evaluate_format(original_examples)

    grader_sampler = ChatCompletionSampler(
        model=resolve_azure_gpt4o_model(grader_model),
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=1000,
        temperature=0,
    )
    eval_class = HealthBenchEval(grader_model=grader_sampler, n_threads=n_threads)
    eval_results = eval_class.evaluate(converted_examples)
    final_result = common.aggregate_results(eval_results)

    input_dir = Path(file_path).parent
    input_name = Path(file_path).stem
    results_path = save_path if save_path is not None else str(input_dir / f"{input_name}_eval_results.json")
    save_evaluation_results(results_path, "healthbench", final_result, converted_examples)

    print("\nEvaluation Summary:")
    print("Task: healthbench")
    print(f"Examples: {len(converted_examples)}")
    print(f"Score: {final_result.score:.3f}")
    print(f"Results saved to: {results_path}")


def evaluate_researchqa(
    file_path: str,
    save_path: str | None,
    grader_model: str,
    n_threads: int,
    batch_size: int,
) -> None:
    input_dir = Path(file_path).parent
    input_name = Path(file_path).stem
    results_path = Path(save_path) if save_path is not None else input_dir / f"{input_name}_eval_results.json"

    original_examples = load_jsonl(file_path)
    response_map = {
        ele["original_data"]["orig_id"]: {
            "answer": ele["final_response"],
        }
        for ele in original_examples
    }

    data_path = download_researchqa_dataset(split="test.json", output_dir=str(AGENT_ROOT / "evaluation" / "research_qa_eval" / "data"))
    results, coverages = compute_coverage(
        data_path=data_path,
        response_map=response_map,
        output_path=None,
        batch_size=batch_size,
        model=resolve_azure_gpt4o_model(grader_model),
        n_threads=n_threads,
    )
    coverage = sum(coverages) / len(coverages) if coverages else 0.0
    aggregated_metrics = _aggregate_researchqa_metrics(data_path, results)

    results_data = {
        "task": "researchqa",
        "run_mode": "evaluation",
        "score": coverage,
        "metrics": {
            "coverage": coverage,
            **aggregated_metrics,
        },
        "metadata": {},
        "num_examples": len(original_examples),
    }
    if results:
        results_data["per_example_results"] = results

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str)

    print("\nEvaluation Summary:")
    print("Task: researchqa")
    print(f"Examples: {len(original_examples)}")
    print(f"Score: {coverage:.3f}")
    print(f"Results saved to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL results file for verl OOD validation.")
    parser.add_argument("task", choices=["healthbench", "researchqa"])
    parser.add_argument("file_path", help="Path to the JSONL results file")
    parser.add_argument("--save_path", default=None, help="Path to save the evaluation results")
    parser.add_argument(
        "--grader-model",
        default="gpt-4o",
        help="Model to use for grading",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=16,
        help="Maximum number of concurrent judge worker threads",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of rubric items grouped into a single ResearchQA judge call",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"File not found: {args.file_path}")
    if not args.file_path.endswith(".jsonl"):
        raise ValueError(f"File must be a JSONL file: {args.file_path}")

    if args.task == "healthbench":
        evaluate_healthbench(args.file_path, args.save_path, args.grader_model, args.n_threads)
    else:
        evaluate_researchqa(
            args.file_path,
            args.save_path,
            grader_model=args.grader_model,
            n_threads=args.n_threads,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
