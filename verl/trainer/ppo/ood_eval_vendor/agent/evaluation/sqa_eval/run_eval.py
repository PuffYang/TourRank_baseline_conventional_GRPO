#!/usr/bin/env python3
"""
Self-contained SQA-CS-V2 evaluation script.

Runs the SQA evaluation pipeline without requiring the external agent-baselines
repository. This script wraps the `astabench` and `inspect_ai` packages.

Usage:
    # Step 1: Convert DR Tulu output to ASTA format
    python run_eval.py convert --input_file <path/to/sqa_output.jsonl>

    # Step 2: Run SQA evaluation on the converted file
    python run_eval.py eval --input_file <path/to/sqa_output_asta_format.jsonl>

    # Or run both steps in one command
    python run_eval.py run --input_file <path/to/sqa_output.jsonl>

    # With options
    python run_eval.py run --input_file <path/to/sqa_output.jsonl> \
        --scorer_model "openai/gpt-4o" \
        --max_connections 16 \
        --split test

Environment Variables Required:
    AZURE_OPENAI_API_KEY: Azure GPT-4o API key
    AZURE_OPENAI_ENDPOINT: Azure GPT-4o endpoint
    OPENAI_API_VERSION: Azure GPT-4o API version
    HF_TOKEN: Hugging Face token (can be dummy if data is local)

Dependencies:
    pip install astabench==0.3.1 inspect_ai datasets
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import DR Tulu format conversion from Varsha's script (single source of truth)
FILE_DIR = Path(__file__).resolve().parent
EVAL_ROOT = FILE_DIR.parent
SQA_DATA_DIR = FILE_DIR / "data"
sys.path.insert(0, str(FILE_DIR))
sys.path.insert(0, str(EVAL_ROOT))
from convert_to_asta_format import parse_answer
from shared_azure_gpt4o import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_ENDPOINT,
    resolve_azure_gpt4o_model,
)


def convert_to_asta_format(input_file: str, output_file: str = None) -> str:
    """
    Convert DR Tulu output to ASTA format for SQA evaluation.

    Uses the DR Tulu citation format: <cite id="xxx">text</cite> in final_response
    with <snippet id="xxx">content</snippet> in tool call traces.

    Args:
        input_file: Path to the DR Tulu output JSONL file
        output_file: Path for the output file (default: input_file with _asta_format suffix)

    Returns:
        Path to the output file
    """
    if output_file is None:
        output_file = input_file.replace(".jsonl", "_asta_format.jsonl")

    print(f"Converting (DR Tulu format): {input_file}")

    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    formatted = []
    for i, sample in enumerate(data):
        try:
            parsed = parse_answer(sample)
            formatted.append(
                {
                    "question": sample["problem"],
                    "response": {"sections": parsed},
                }
            )
        except Exception as e:
            print(f"Warning: Failed to convert sample {i}: {e}")

    with open(output_file, "w") as f:
        json.dump(formatted, f, indent=4)

    print(f"Converted {len(formatted)}/{len(data)} samples -> {output_file}")
    return output_file


# ============================================================================
# SQA Evaluation (wrapping astabench/inspect)
# ============================================================================
def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import astabench
    except ImportError:
        missing.append("astabench==0.3.1")

    try:
        import inspect_ai
    except ImportError:
        missing.append("inspect_ai")

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def create_cached_solver_script(data_path: str, split: str = "test") -> str:
    """
    Create a temporary solver script that loads cached results.

    This replicates the behavior of agent_baselines/solvers/sqa/debug/cached_solver.py
    without needing the agent-baselines repo.
    """
    solver_code = f'''
import json
from pathlib import Path

from astabench.types.sqa import SQAResponse, SQAResponseWithUsage
from astabench.util.model import record_model_usage_with_inspect
from inspect_ai.model import ChatMessageAssistant, ModelUsage
from inspect_ai.solver import Generate, Solver, TaskState, solver
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()
import os
from datasets import load_dataset

_dataset = None

def _load_dataset(path: str, split: str):
    global _dataset
    if _dataset is None:
        if os.path.exists(path):
            print(f"Loading data from local file: {{path}}")
            _dataset = load_dataset("json", data_files={{split: path}})[split]
        else:
            raise FileNotFoundError(f"Data file not found: {{path}}")

def _read_q_result(question: str):
    results = _dataset.filter(lambda x: x["question"] == question)
    if len(results) == 0:
        return None
    return results[0]["response"]

def _query_cache(question: str):
    response_dict = _read_q_result(question)
    if response_dict is None:
        raise ValueError(f"No results found for question: {{question}}")
    response = (
        SQAResponseWithUsage(**response_dict)
        if "tokens" in response_dict
        else SQAResponse(**response_dict)
    )
    return response

@solver
def cache_solver(
    path: str = "{data_path}",
    split: str = "{split}",
    model: str = "openai/gpt-4o",
) -> Solver:
    _load_dataset(path, split)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata["initial_prompt"]
        response = _query_cache(question)
        content = json.dumps(
            {{"response": response.model_dump(mode="json", exclude={{"tokens"}})}},
            indent=2,
        )
        if hasattr(response, "tokens"):
            usage_dict = {{f"{{k}}_tokens": v for k, v in response.tokens.items()}}
            record_model_usage_with_inspect(
                model_name=model,
                usage=ModelUsage(**usage_dict),
            )
        state.messages.append(ChatMessageAssistant(content=content))
        state.output.completion = content
        return state

    return solve
'''
    return solver_code


def _normalize_sqa_scorer_model(scorer_model: str) -> str:
    resolved = resolve_azure_gpt4o_model(scorer_model)
    return resolved if "/" in resolved else f"openai/{resolved}"


def _resolve_local_sqa_rubrics_dir() -> Path:
    override_dir = os.environ.get("SQA_RUBRICS_DATA_DIR")
    if override_dir:
        return Path(override_dir).expanduser()
    return SQA_DATA_DIR


def _expected_sqa_rubrics_filename(split: str) -> str:
    version = "v2" if split == "test" else "v1"
    return f"rubrics_{version}_recomputed.json"


def _has_local_sqa_rubrics(split: str) -> bool:
    expected = _resolve_local_sqa_rubrics_dir() / "tasks" / "sqa" / _expected_sqa_rubrics_filename(split)
    return expected.exists()


def _build_openai_sitecustomize() -> str:
    return f'''
import os
import shutil
from pathlib import Path

DEFAULT_ENDPOINT = {DEFAULT_AZURE_OPENAI_ENDPOINT!r}
DEFAULT_API_VERSION = {DEFAULT_AZURE_OPENAI_API_VERSION!r}
DEFAULT_SQA_RUBRICS_DIR = {str(_resolve_local_sqa_rubrics_dir())!r}

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", DEFAULT_ENDPOINT)
os.environ.setdefault("OPENAI_API_VERSION", DEFAULT_API_VERSION)
if os.environ.get("AZURE_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

try:
    import huggingface_hub

    _original_hf_hub_download = huggingface_hub.hf_hub_download

    def _patched_hf_hub_download(repo_id, filename, *args, **kwargs):
        local_root = Path(os.environ.get("SQA_RUBRICS_DATA_DIR", DEFAULT_SQA_RUBRICS_DIR)).expanduser()
        local_path = local_root / filename
        if repo_id == "allenai/asta-bench" and local_path.exists():
            return str(local_path)
        downloaded = _original_hf_hub_download(repo_id, filename, *args, **kwargs)
        if (
            repo_id == "allenai/asta-bench"
            and os.environ.get("SQA_DOWNLOAD_MISSING_DATA", "0") == "1"
        ):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                shutil.copy2(downloaded, local_path)
            return str(local_path)
        return downloaded

    huggingface_hub.hf_hub_download = _patched_hf_hub_download
except Exception as exc:
    print(f"WARNING: Failed to patch hf_hub_download for local SQA rubrics: {{exc}}")

try:
    import openai
    from openai import AsyncAzureOpenAI, AzureOpenAI

    class _PatchedOpenAI(AzureOpenAI):
        def __init__(self, *args, **kwargs):
            kwargs.pop("base_url", None)
            kwargs.pop("organization", None)
            kwargs.pop("project", None)
            kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            kwargs["api_version"] = os.environ.get("OPENAI_API_VERSION", DEFAULT_API_VERSION)
            kwargs["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_ENDPOINT)
            super().__init__(*args, **kwargs)

    class _PatchedAsyncOpenAI(AsyncAzureOpenAI):
        def __init__(self, *args, **kwargs):
            kwargs.pop("base_url", None)
            kwargs.pop("organization", None)
            kwargs.pop("project", None)
            kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            kwargs["api_version"] = os.environ.get("OPENAI_API_VERSION", DEFAULT_API_VERSION)
            kwargs["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_ENDPOINT)
            super().__init__(*args, **kwargs)

    openai.OpenAI = _PatchedOpenAI
    openai.AsyncOpenAI = _PatchedAsyncOpenAI
except Exception as exc:
    print(f"WARNING: Failed to patch OpenAI client to Azure GPT-4o: {{exc}}")
'''


def _build_inspect_cmd(
    solver_path: str,
    input_file: str,
    split: str,
    scorer_model: str,
    max_connections: int,
    simplified_eval: bool,
    assess_jointly: bool,
    sentence_wise_cit_eval: bool,
    all_at_once: bool,
    with_search_tools: bool,
    output_dir: str = None,
) -> List[str]:
    """Build the inspect eval command arguments (without the python/uv prefix)."""
    args = [
        "eval", "astabench/sqa",
        "--display", "plain",
        "--solver", solver_path,
        "-S", f"path={os.path.abspath(input_file)}",
        "-T", f"split={split}",
        "-T", f"with_search_tools={with_search_tools}",
        "-T", f"simplified_eval={str(simplified_eval).lower()}",
        "-T", f"assess_jointly={str(assess_jointly).lower()}",
        "--max-connections", str(max_connections),
        "-T", f"sentence_wise_cit_eval={str(sentence_wise_cit_eval).lower()}",
        "-T", f"all_at_once={str(all_at_once).lower()}",
        "-T", f"scorer_model={scorer_model}",
    ]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        args.extend(["--log-dir", output_dir])
    return args


def _check_uv_available() -> bool:
    """Check if uv is available."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _collect_numeric_metrics(obj: Any, targets: set[str], found: Dict[str, float]) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in targets and isinstance(val, (int, float)) and key not in found:
                found[key] = float(val)
            _collect_numeric_metrics(val, targets, found)
        return

    if isinstance(obj, list):
        for item in obj:
            _collect_numeric_metrics(item, targets, found)


def _extract_metrics_from_output_dir(output_dir: str) -> Dict[str, float]:
    targets = {
        "global_avg",
        "ingredient_recall",
        "answer_precision",
        "citation_recall",
        "citation_precision",
    }
    metrics: Dict[str, float] = {}
    root = Path(output_dir)
    if not root.exists():
        return metrics

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".json", ".jsonl"} or not path.is_file():
            continue

        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as f:
                    _collect_numeric_metrics(json.load(f), targets, metrics)
            else:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        _collect_numeric_metrics(json.loads(line), targets, metrics)
        except Exception:
            continue

    return metrics


def _extract_metrics_from_text(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in [
        "global_avg",
        "ingredient_recall",
        "answer_precision",
        "citation_recall",
        "citation_precision",
    ]:
        match = re.search(rf"{key}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def run_sqa_evaluation(
    input_file: str,
    split: str = "test",
    scorer_model: str = "openai/gpt-4o",
    max_connections: int = 16,
    simplified_eval: bool = True,
    assess_jointly: bool = True,
    sentence_wise_cit_eval: bool = False,
    all_at_once: bool = True,
    with_search_tools: bool = False,
    output_dir: str = None,
):
    """
    Run SQA evaluation using astabench + inspect_ai.

    Automatically handles the Python 3.11+ requirement:
    - If current Python >= 3.11 and packages are installed: runs directly
    - Otherwise: uses `uv run` with a temporary project to manage dependencies

    Args:
        input_file: Path to the ASTA-formatted JSONL file
        split: Dataset split to evaluate on (test/dev)
        scorer_model: Model to use for scoring
        max_connections: Maximum concurrent connections
        simplified_eval: Use simplified evaluation
        assess_jointly: Assess jointly
        sentence_wise_cit_eval: Sentence-wise citation evaluation
        all_at_once: Process all at once
        with_search_tools: Whether to use search tools
        output_dir: Directory for evaluation output
    """
    scorer_model = _normalize_sqa_scorer_model(scorer_model)

    if not (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("WARNING: AZURE_OPENAI_API_KEY not set. Required for Azure GPT-4o scoring.")

    # Create temporary solver script
    solver_code = create_cached_solver_script(
        data_path=os.path.abspath(input_file), split=split
    )

    # Write to temporary directory
    solver_dir = tempfile.mkdtemp(prefix="sqa_eval_")
    solver_path = os.path.join(solver_dir, "cached_solver.py")
    with open(solver_path, "w") as f:
        f.write(solver_code)
    sitecustomize_path = os.path.join(solver_dir, "sitecustomize.py")
    with open(sitecustomize_path, "w") as f:
        f.write(_build_openai_sitecustomize())

    # Build inspect eval arguments
    inspect_args = _build_inspect_cmd(
        solver_path=solver_path,
        input_file=input_file,
        split=split,
        scorer_model=scorer_model,
        max_connections=max_connections,
        simplified_eval=simplified_eval,
        assess_jointly=assess_jointly,
        sentence_wise_cit_eval=sentence_wise_cit_eval,
        all_at_once=all_at_once,
        with_search_tools=with_search_tools,
        output_dir=output_dir,
    )

    # Determine execution method
    use_uv = False
    py_version = sys.version_info

    if py_version >= (3, 11) and check_dependencies():
        # Direct execution: Python >= 3.11 and deps installed
        cmd = [sys.executable, "-m", "inspect_ai._cli.main"] + inspect_args
    elif _check_uv_available():
        # Use uv run: manages Python version + deps automatically
        use_uv = True
        # Create a temporary pyproject.toml for uv
        pyproject_content = """
[project]
name = "sqa-eval-runner"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "astabench==0.3.1",
    "inspect_ai",
    "datasets",
]

[tool.uv]
override-dependencies = [
    "openai==1.78.0",
]
"""
        pyproject_path = os.path.join(solver_dir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write(pyproject_content)

        cmd = [
            "uv", "run", "--python", "3.11",
            "inspect",
        ] + inspect_args
    else:
        print("ERROR: Cannot run SQA evaluation:")
        print(f"  - Current Python: {py_version.major}.{py_version.minor} (need >= 3.11)")
        print(f"  - astabench requires Python >= 3.11")
        print()
        print("Options:")
        print("  1. Install uv: pip install uv")
        print("  2. Use Python >= 3.11 with: pip install astabench==0.3.1 inspect_ai datasets")
        print("  3. Create a conda env: conda create -n sqa_eval python=3.11 && conda activate sqa_eval && pip install astabench==0.3.1 inspect_ai datasets")
        # Cleanup
        try:
            import shutil
            shutil.rmtree(solver_dir)
        except Exception:
            pass
        sys.exit(1)

    print(f"\n=== Running SQA Evaluation ===")
    print(f"Input file: {input_file}")
    print(f"Split: {split}")
    print(f"Scorer model: {scorer_model}")
    print(f"Max connections: {max_connections}")
    print(f"Method: {'uv run (auto-managed Python 3.11+)' if use_uv else 'direct'}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Pass through environment variables and redirect OpenAI clients to Azure GPT-4o
    env = os.environ.copy()
    if env.get("AZURE_OPENAI_API_KEY") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["AZURE_OPENAI_API_KEY"]
    env.setdefault("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_OPENAI_ENDPOINT)
    env.setdefault("OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION)
    env.setdefault("SQA_RUBRICS_DATA_DIR", str(_resolve_local_sqa_rubrics_dir()))
    env["PYTHONPATH"] = (
        solver_dir
        if not env.get("PYTHONPATH")
        else f"{solver_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    has_local_rubrics = _has_local_sqa_rubrics(split)
    # HF_TOKEN is only needed when local SQA rubrics are unavailable.
    if not env.get("HF_TOKEN") and not has_local_rubrics:
        # Try to get token from huggingface-cli login cache
        try:
            from huggingface_hub import HfFolder
            cached_token = HfFolder.get_token()
            if cached_token:
                env["HF_TOKEN"] = cached_token
                print(f"Using cached HF token from huggingface-cli login")
        except Exception:
            pass
    if not env.get("HF_TOKEN") and not has_local_rubrics:
        print("WARNING: HF_TOKEN not set. Needed for gated allenai/asta-bench dataset.")
        print("  Run: huggingface-cli login")
        print("  Or set: export HF_TOKEN='your_token'")
        print("  And request access at: https://huggingface.co/datasets/allenai/asta-bench")
    elif has_local_rubrics:
        print(f"Using local SQA rubrics from: {_resolve_local_sqa_rubrics_dir()}")

    # Run the evaluation
    result = subprocess.run(cmd, cwd=solver_dir, env=env, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Cleanup
    try:
        import shutil
        shutil.rmtree(solver_dir)
    except Exception:
        pass

    if result.returncode != 0:
        print(f"\nEvaluation failed with return code {result.returncode}")
        sys.exit(result.returncode)
    else:
        metrics = {}
        if output_dir:
            metrics.update(_extract_metrics_from_output_dir(output_dir))
        metrics.update({k: v for k, v in _extract_metrics_from_text(result.stdout or "").items() if k not in metrics})

        if output_dir:
            summary_path = Path(output_dir) / "summary.json"
            summary = {
                "score": metrics.get("global_avg"),
                "metrics": metrics,
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n=== SQA Evaluation Complete ===")
        if output_dir:
            print(f"Logs saved to: {output_dir}")


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Self-contained SQA-CS-V2 Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert DR Tulu output to ASTA format:
  python run_eval.py convert --input_file eval_output/sqa.jsonl

  # Run evaluation on converted file:
  python run_eval.py eval --input_file eval_output/sqa_asta_format.jsonl

  # Full pipeline (convert + evaluate):
  python run_eval.py run --input_file eval_output/sqa.jsonl

  # With custom scorer model:
  python run_eval.py run --input_file eval_output/sqa.jsonl \\
      --scorer_model "openai/gpt-4o"
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert DR Tulu output to ASTA format"
    )
    convert_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the DR Tulu output JSONL file",
    )
    convert_parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path for the output file (default: auto-generated)",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval", help="Run SQA evaluation on ASTA-formatted file"
    )
    eval_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the ASTA-formatted JSONL file",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    eval_parser.add_argument(
        "--scorer_model",
        type=str,
        default="openai/gpt-4o",
        help="Scorer model (default: openai/gpt-4o)",
    )
    eval_parser.add_argument(
        "--max_connections",
        type=int,
        default=16,
        help="Maximum concurrent connections (default: 16)",
    )
    eval_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation output logs",
    )

    # Run command (convert + eval)
    run_parser = subparsers.add_parser(
        "run", help="Full pipeline: convert + evaluate"
    )
    run_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the DR Tulu output JSONL file",
    )
    run_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    run_parser.add_argument(
        "--scorer_model",
        type=str,
        default="openai/gpt-4o",
        help="Scorer model (default: openai/gpt-4o)",
    )
    run_parser.add_argument(
        "--max_connections",
        type=int,
        default=16,
        help="Maximum concurrent connections (default: 16)",
    )
    run_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation output logs",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "convert":
        convert_to_asta_format(args.input_file, args.output_file)

    elif args.command == "eval":
        run_sqa_evaluation(
            input_file=args.input_file,
            split=args.split,
            scorer_model=args.scorer_model,
            max_connections=args.max_connections,
            output_dir=args.output_dir,
        )

    elif args.command == "run":
        # Step 1: Convert
        print("=== Step 1: Format Conversion ===")
        asta_file = convert_to_asta_format(args.input_file)

        # Step 2: Evaluate
        print("\n=== Step 2: SQA Evaluation ===")
        run_sqa_evaluation(
            input_file=asta_file,
            split=args.split,
            scorer_model=args.scorer_model,
            max_connections=args.max_connections,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
