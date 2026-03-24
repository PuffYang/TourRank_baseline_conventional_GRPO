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
"""
Build verl RL parquet datasets from TourRank query_rubrics.jsonl.
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _normalize_rubrics(rubrics):
    if not isinstance(rubrics, list):
        return []

    normalized = []
    for item in rubrics:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        weight = item.get("weight", 1)
        try:
            weight = int(weight)
        except Exception:
            weight = 1
        if not title or not description:
            continue
        normalized.append({"title": title, "description": description, "weight": weight})
    return normalized


def _load_rows(input_jsonl: Path, data_source: str, ability: str):
    rows = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = str(obj.get("query", "")).strip()
            rubrics = _normalize_rubrics(obj.get("rubrics", []))
            if not query or not rubrics:
                continue

            row = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": query}],
                "ability": ability,
                "reward_model": {
                    "style": "rubric_judge",
                    "ground_truth": {"query": query, "rubrics": rubrics},
                },
                "extra_info": {
                    "index": idx,
                    "query": query,
                    "rubrics": rubrics,
                },
            }
            rows.append(row)
    return rows


def _split_rows(rows, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return rows, rows
    n_total = len(rows)
    n_val = max(1, int(n_total * val_ratio))
    rng = random.Random(seed)
    all_indices = list(range(n_total))
    rng.shuffle(all_indices)
    val_idx = set(all_indices[:n_val])

    train_rows = [row for i, row in enumerate(rows) if i not in val_idx]
    val_rows = [row for i, row in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser(description="Build TourRank rubric parquet dataset for verl GRPO.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/Users/yangzixuan/Desktop/TourRank/DR-Tulu/data/RL/query_rubrics.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yangzixuan/Desktop/TourRank/DR-Tulu/data/RL/verl_rubric_grpo"),
    )
    parser.add_argument("--train-name", type=str, default="train.parquet")
    parser.add_argument("--val-name", type=str, default="val.parquet")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation split ratio. 0 means val=train (same file content).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", type=str, default="tourrank/query_rubrics")
    parser.add_argument("--ability", type=str, default="tourrank")
    args = parser.parse_args()

    rows = _load_rows(args.input_jsonl, args.data_source, args.ability)
    if len(rows) == 0:
        raise ValueError(f"No valid rows loaded from {args.input_jsonl}")

    train_rows, val_rows = _split_rows(rows, args.val_ratio, args.seed)
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError(f"Invalid split result: train={len(train_rows)} val={len(val_rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name

    train_table = pa.Table.from_pylist(train_rows)
    val_table = pa.Table.from_pylist(val_rows)
    pq.write_table(train_table, train_path)
    pq.write_table(val_table, val_path)

    print(
        f"Done. input={args.input_jsonl} total={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"train_path={train_path} val_path={val_path}"
    )


if __name__ == "__main__":
    main()
